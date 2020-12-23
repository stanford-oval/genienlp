#
# Copyright (c) 2018, Salesforce, Inc.
#                     The Board of Trustees of the Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import logging
import logging.handlers
import math
import os
import time
from copy import deepcopy
from functools import partial
from pprint import pformat

import numpy as np
import torch
from tensorboardX import SummaryWriter
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, AdamW

from . import arguments
from . import models
from .util import elapsed_time, set_seed, get_trainable_params, make_data_loader,\
    log_model_size, init_devices
from .model_utils.parallel_utils import NamedTupleCompatibleDataParallel
from .model_utils.saver import Saver
from .validate import validate


def initialize_logger(args):
    # set up file logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(os.path.join(args.log_dir, f'train.log'),
                                                   maxBytes=1024 * 1024 * 10, backupCount=1)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def prepare_data(args, logger):
    train_sets, val_sets, aux_sets = [], [], []
    for task in args.train_tasks:
        logger.info(f'Loading {task.name}')
        kwargs = {'test': None, 'validation': None}
        kwargs.update({'subsample': args.subsample, 'skip_cache': args.skip_cache, 'cache_input_data': args.cache_input_data,
                       'cached_path': os.path.join(args.cache, task.name), 'all_dirs': args.train_languages,
                       'sentence_batching': args.sentence_batching, 'almond_lang_as_question': args.almond_lang_as_question})
        if args.use_curriculum:
            kwargs['curriculum'] = True

        logger.info(f'Adding {task.name} to training datasets')
        split = task.get_splits(args.data, lower=args.lower, **kwargs)
        assert not split.eval and not split.test
        if args.use_curriculum:
            assert split.aux
            aux_sets.append(split.aux)
            logger.info(f'{task.name} has {len(split.aux)} auxiliary examples')
        else:
            assert split.train
        train_sets.append(split.train)
        logger.info(f'{task.name} has {len(split.train)} training examples')

    for task in args.val_tasks:
        logger.info(f'Loading {task.name}')
        kwargs = {'train': None, 'test': None}
        # choose best model based on this dev set
        if args.eval_set_name is not None:
            kwargs['validation'] = args.eval_set_name
        kwargs.update({'subsample': args.subsample, 'skip_cache': args.skip_cache, 'cache_input_data': args.cache_input_data,
                       'cached_path': os.path.join(args.cache, task.name), 'all_dirs': args.eval_languages,
                        'almond_lang_as_question': args.almond_lang_as_question})
        
        logger.info(f'Adding {task.name} to validation datasets')
        split = task.get_splits(args.data, lower=args.lower, **kwargs)
        assert not split.train and not split.test and not split.aux
        logger.info(f'{task.name} has {len(split.eval)} validation examples')
        val_sets.append(split.eval)

    return train_sets, val_sets, aux_sets


accumulated_batch_lengths = 0


def train_step(model, batch, iteration, opt, devices, lr_scheduler=None, grad_clip=None,
               gradient_accumulation_steps=1):
    # Since the batch size is different in each call to this function due to dynamic batching, we need to keep track of
    # the total batch size
    global accumulated_batch_lengths
    model.train()
    if (iteration) % gradient_accumulation_steps == 0:
        opt.zero_grad()
    loss = model(batch).loss
    if torch.isnan(loss).any():
        raise RuntimeError('Got NaN loss %s', str(loss))
    if len(devices) > 1:
        loss = loss.mean()
    non_accumulated_loss = loss.item()
    loss = loss*len(batch[0])
    accumulated_batch_lengths += len(batch[0])

    loss.backward()
    grad_norm = None
    if (iteration+1) % gradient_accumulation_steps == 0:
        for p in model.parameters():
            if p.grad is None:
                continue
            p.grad /= accumulated_batch_lengths
        accumulated_batch_lengths = 0
        if grad_clip > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.params, grad_clip)
        opt.step()
        lr_scheduler.step()

    return non_accumulated_loss, grad_norm


def update_fraction(args, task_iteration):
    if args.curriculum_strategy == 'linear':
        next_fraction = args.curriculum_rate * task_iteration
    elif args.curriculum_strategy == 'exp':
        next_fraction = args.curriculum_rate * np.exp(task_iteration)

    fraction = min(args.curriculum_max_frac, next_fraction)

    return fraction


def should_validate(iteration, val_every, resume, start_iteration):
    if val_every is None:
        return False
    return (iteration % val_every == 0) or (resume and iteration == start_iteration)


def should_save(iteration, save_every):
    if save_every is None:
        return False
    return iteration % save_every == 0


def should_log(iteration, log_every):
    if log_every is None:
        return False
    return iteration % log_every == 0


def do_validate(iteration, args, model, numericalizer, val_iters, *,
                train_task, round_progress, task_progress, writer, logger):
    deca_score = 0
    for val_task_idx, (val_task, val_iter) in enumerate(val_iters):
        val_loss, metric_dict = validate(val_task, val_iter, model, numericalizer, args, num_print=args.num_print)
        if val_loss is not None:
            log_entry = f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{train_task.name}:{task_progress}val_{val_task.name}:val_loss{val_loss.item():.4f}:'
            writer.add_scalar(f'loss/{val_task.name}/val', val_loss.item(), iteration)
        else:
            log_entry = f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{train_task.name}:{task_progress}val_{val_task.name}:'

        metric_entry = ''
        for metric_key, metric_value in metric_dict.items():
            metric_entry += f'{metric_key}_{metric_value:.2f}:'
        metric_entry = metric_entry[:-1]

        deca_score += metric_dict[val_task.metrics[0]]

        # val log
        logger.info(log_entry + metric_entry)
        if writer is not None:
            for metric_key, metric_value in metric_dict.items():
                writer.add_scalar(f'{val_task.name}/{metric_key}/val', metric_value, iteration)
    if writer is not None:
        writer.add_scalar('deca/val', deca_score, iteration)
    logger.info(
        f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{train_task.name}:{task_progress}val_deca:deca_{deca_score:.2f}')

    return deca_score


def maybe_save(iteration, model, opt, deca_score, best_decascore, *,
               saver, logger, train_task, round_progress, task_progress, timestamp, log_dir):
    should_save_best = False
    if deca_score is not None and (best_decascore is None or best_decascore < deca_score):
        best_decascore = deca_score
        should_save_best = True

    # punch through the nn.DataParallel to access the real model, otherwise we won't be able
    # to load this model later
    model_state_dict = model.module.state_dict()
    model_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}

    save_model_state_dict = {
        'model_state_dict': model_state_dict,
        'best_decascore': best_decascore
    }
    save_opt_state_dict = opt.state_dict()
    save_opt_state_dict.update({'start_iteration': iteration})

    saver.save(save_model_state_dict, save_opt_state_dict, global_step=iteration)
    if should_save_best:
        logger.info(
            f'{timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{train_task.name}:{task_progress}found new best model')
        torch.save(save_model_state_dict, os.path.join(log_dir, 'best.pth'))
        torch.save(save_opt_state_dict, os.path.join(log_dir, 'best_optim.pth'))
        model.module.numericalizer.save(saver._savedir)

    return best_decascore


def do_log_training_loss(iteration, loss, *,
                         lr_scheduler, grad_norm,
                         num_examples, len_contexts, len_answers,
                         logger, train_task, round_progress, task_progress,
                         timestamp, writer, log_prefix):
    avg_batch_size = f'avbatch_{num_examples:.0f}_{len_contexts:.0f}_{len_answers:.0f}:'
    logger.info(
        f'{timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{train_task.name}:{task_progress}{avg_batch_size}{log_prefix}/loss_{loss:.4f}')

    if writer is not None:
        writer.add_scalar(f'{log_prefix}/loss/{train_task.name}', loss, iteration)

        if lr_scheduler is not None:
            writer.add_scalar(f'{log_prefix}/lr', lr_scheduler.get_last_lr(), iteration)
        if grad_norm is not None:
            writer.add_scalar(f'{log_prefix}/norm', grad_norm, iteration)


def np_coin(prob):
    return np.random.uniform() < prob


def get_next_batch(train_iter, aux_iters, *, task, task_idx, task_fraction, use_curriculum):
    if use_curriculum:
        aux_iter = aux_iters[task_idx][1]
        prob = np_coin(task_fraction[task])
        if prob == 'aux':
            return next(aux_iter)
        else:
            assert prob == 'train'
            batch = next(train_iter)

    else:
        batch = next(train_iter)

    return batch


def train(args, devices, model, opt, lr_scheduler, train_sets, train_iterations, numericalizer, *,
          log_every, val_every, save_every, rounds, val_sets, aux_sets, writer, logger, log_prefix,
          start_iteration=1, rnd=1, best_decascore, use_curriculum):
    """main training function"""
    local_loss, num_examples, len_contexts, len_answers, iteration = 0, 0, 0, 0, start_iteration

    train_iter_deep = deepcopy(train_iterations)

    task_iteration = dict()
    task_done = dict()
    task_fraction = dict()

    for task in args.train_tasks:
        task_iteration[task] = 1
        task_done[task] = False
        task_fraction[task] = 0.0

    saver = Saver(args.log_dir, args.max_to_keep)
    epoch = 0

    logger.info(f'Preparing iterators')
    main_device = devices[0]
    train_iters = [(task, make_data_loader(x, numericalizer, tok, main_device, train=True))
                   for task, x, tok in zip(args.train_tasks, train_sets, args.train_batch_tokens)]
    train_iters = [(task, iter(train_iter)) for task, train_iter in train_iters]

    val_iters = [(task, make_data_loader(x, numericalizer, bs, main_device, train=False))
                 for task, x, bs in zip(args.val_tasks, val_sets, args.val_batch_size)]

    aux_iters = []
    if use_curriculum:
        aux_iters = [(name, make_data_loader(x, numericalizer, tok, main_device, train=True))
                     for name, x, tok in zip(args.train_tasks, aux_sets, args.train_batch_tokens)]
        aux_iters = [(task, iter(aux_iter)) for task, aux_iter in aux_iters]
        
    zero_loss = 0
    logger.info(f'Begin {log_prefix}')

    while not all(task_done.values()):
        # For some number of rounds, we 'jump start' some subset of the tasks
        # by training them and not others
        # once the specified number of rounds is completed,
        # switch to normal round robin training
        if rnd < args.jump_start:
            train_iterations = [0] * len(train_iterations)
            for j in range(args.n_jump_start): train_iterations[j] = 1
        else:
            train_iterations = train_iter_deep

        for task_idx, (task, train_iter) in enumerate(train_iters):
            task_iterations = train_iterations[task_idx] if train_iterations is not None else None
            if task_iterations == 0:
                continue
            if task_iterations is not None and task_iteration[task] > task_iterations:
                task_done[task] = True
                continue

            batch = get_next_batch(train_iter, aux_iters, task=task, task_idx=task_idx,
                                   task_fraction=task_fraction, use_curriculum=use_curriculum)

            if iteration < start_iteration:
                # skip this iteration (this is done to ensure iterators are at the same position when resuming)
                task_iteration[task] += 1
                iteration += 1
                return

            task_progress = f'{task_iteration[task]}/{task_iterations}:' if task_iterations is not None else ''
            round_progress = f'round_{rnd}:' if rounds else ''

            # validate
            if should_validate(iteration, val_every, resume=args.resume, start_iteration=start_iteration):
                deca_score = do_validate(iteration, args, model, numericalizer, val_iters,
                                         train_task=task, round_progress=round_progress,
                                         task_progress=task_progress, writer=writer, logger=logger)

                # saving
                if should_save(iteration, save_every):
                    best_decascore = maybe_save(iteration, model, opt, deca_score, best_decascore,
                                                saver=saver, logger=logger, train_task=task,
                                                round_progress=round_progress, task_progress=task_progress,
                                                timestamp=args.timestamp, log_dir=args.log_dir)

            # param update
            loss, grad_norm = train_step(model, batch, iteration, opt, devices, lr_scheduler=lr_scheduler,
                                         grad_clip=args.grad_clip,
                                         gradient_accumulation_steps=args.gradient_accumulation_steps)
            if loss is None:
                logger.info('Encountered NAN loss during training... Continue training ignoring the current batch')
                continue
            if loss < 1e-6:
                zero_loss += 1
                if zero_loss >= 100:
                    logger.info('Found loss less than 1e-5 for 100 steps, stopping.')
                    return
            else:
                zero_loss = 0

            # update curriculum fraction
            if args.use_curriculum:
                task_fraction[task] = update_fraction(args, task_iteration[task])

            # train metrics
            local_loss += loss

            # train logs
            num_examples += batch.context.value.size(0)
            len_contexts += batch.context.value.size(1)
            len_answers += batch.answer.value.size(1)

            if should_log(iteration, log_every):
                local_loss /= log_every
                num_examples /= log_every
                len_contexts /= log_every
                len_answers /= log_every
                do_log_training_loss(iteration, local_loss,
                                     lr_scheduler=lr_scheduler, grad_norm=grad_norm,
                                     num_examples=num_examples, len_contexts=len_contexts, len_answers=len_answers,
                                     logger=logger, writer=writer, train_task=task, round_progress=round_progress,
                                     task_progress=task_progress, timestamp=args.timestamp, log_prefix=log_prefix)
                num_examples = 0
                len_contexts = 0
                len_answers = 0
                local_loss = 0

            # book keeping
            task_iteration[task] += 1
            iteration += 1

        # book keeping
        epoch += 1
        rnd += 1

    logger.info(f'{log_prefix} is done after {epoch} epochs')



def get_transformer_learning_rate(i, *, dimension, warmup):
    i += 1
    return 1. / math.sqrt(dimension) * min(1 / math.sqrt(i), i / (warmup * math.sqrt(warmup)))


def get_sgd_learning_rate(i, *, warmup):
    i += 1
    return min(math.sqrt(warmup) / math.sqrt(i), i / warmup)


def init_opt(args, model, logger):
    if args.optimizer == 'adam':
        # Adam with transformer schedule has a different set of default hyperparameters:
        if args.lr_schedule == 'transformer':
            opt = torch.optim.Adam(model.params, lr=args.lr_multiply, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
        else:
            opt = torch.optim.Adam(model.params, lr=args.lr_multiply, betas=(args.beta0, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        opt = AdamW(model.params, lr=args.lr_multiply, weight_decay=args.weight_decay)
    elif args.optimizer == 'radam':
        import radam
        if args.warmup > 1:
            logger.warning('With RAdam optimizer, warmup is never applied')
        opt = radam.RAdam(model.params, lr=args.lr_multiply, betas=(args.beta0, 0.999), weight_decay=args.weight_decay)
    else:
        assert args.optimizer == 'sgd'
        opt = torch.optim.SGD(model.params, lr=args.lr_multiply, weight_decay=args.weight_decay)
    
    if args.lr_schedule == 'transformer':
        lr_lambda = partial(get_transformer_learning_rate, dimension=args.dimension, warmup=args.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    elif args.lr_schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(opt, num_training_steps=sum(args.train_iterations), num_warmup_steps=args.warmup)
    elif args.lr_schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(opt, num_training_steps=sum(args.train_iterations), num_warmup_steps=args.warmup)
    elif args.lr_schedule == 'sgd':
        lr_lambda = partial(get_sgd_learning_rate, warmup=args.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        raise ValueError('Invalid learning rate scheduler.')
    

    return opt, scheduler


def main(args):
    args = arguments.post_parse(args)
    if args is None:
        return

    set_seed(args)
    devices = init_devices(args, args.devices)
    logger = initialize_logger(args)
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    model_name = args.model
    model_class = getattr(models, model_name)

    tasks = set(args.train_tasks) | set(args.val_tasks)
    train_sets, val_sets, aux_sets = prepare_data(args, logger)

    if (args.use_curriculum and aux_sets is None) or (not args.use_curriculum and len(aux_sets) > 0):
        logging.error('Something unpleasant is happening with curriculum')

    logger.info(f'Processing')
    logger.start = time.time()

    ########## initialize model
    best_decascore = None
    if args.load is not None:
        model, best_decascore = model_class.from_pretrained(args.save,
                                                            args=args,
                                                            model_checkpoint_file=args.load,
                                                            vocab_sets=train_sets+val_sets,
                                                            tasks=tasks,
                                                            device=devices[0])
        model.add_new_vocab_from_data(tasks=tasks, resize_decoder=True)
    else:
        logger.info(f'Initializing a new {model_name}')
        model = model_class(args=args, vocab_sets=train_sets+val_sets, tasks=tasks)

    params = get_trainable_params(model)
    log_model_size(logger, model, model_name)

    model.to(devices[0])
    model = NamedTupleCompatibleDataParallel(model, device_ids=devices)
    model.params = params
    ##########

    opt, lr_scheduler = init_opt(args, model, logger)
    start_iteration = 1

    if args.resume:
        logger.info(f'Resuming training from {os.path.splitext(args.load)[0]}_optim.pth')
        opt_state_dict = torch.load(os.path.join(args.save, f'{os.path.splitext(args.load)[0]}_optim.pth'))
        start_iteration = opt_state_dict.pop('start_iteration')
        logger.info(f'Starting iteration is {start_iteration}')
        opt.load_state_dict(opt_state_dict)

    if hasattr(args, 'tensorboard') and args.tensorboard:
        logger.info(f'Initializing Writer')
        writer = SummaryWriter(log_dir=args.tensorboard_dir, purge_step=start_iteration)
    else:
        writer = None

    train(args, devices, model, opt, lr_scheduler, train_sets,
          args.train_iterations, model.module.numericalizer, val_sets=val_sets, aux_sets=aux_sets, logger=logger, writer=writer,
          log_every=args.log_every, val_every=args.val_every, save_every=args.save_every,
          rounds=len(train_sets) > 1, start_iteration=start_iteration, use_curriculum=args.use_curriculum,
          best_decascore=best_decascore, log_prefix='training')
