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

from . import arguments
from . import models
from .data_utils.embeddings import load_embeddings
from .data_utils.example import Example
from .util import elapsed_time, set_seed, preprocess_examples, get_trainable_params, make_data_loader, log_model_size, \
    init_devices
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
    train_sets, val_sets, aux_sets, vocab_sets = [], [], [], []
    for task in args.train_tasks:
        logger.info(f'Loading {task.name}')
        kwargs = {'test': None}
        kwargs['subsample'] = args.subsample
        kwargs['validation'] = None
        if args.use_curriculum:
            kwargs['curriculum'] = True
        kwargs['skip_cache'] = args.skip_cache
        kwargs['cached_path'] = os.path.join(args.cache, task.name)

        logger.info(f'Adding {task.name} to training datasets')
        split = task.get_splits(args.data, lower=args.lower, **kwargs)
        if args.use_curriculum:
            assert len(split) == 2
            aux_sets.append(split[1])
            logger.info(f'{task.name} has {len(split[1])} auxiliary examples')
        else:
            assert len(split) == 1
        train_sets.append(split[0])
        logger.info(f'{task.name} has {len(split[0])} training examples')
        if args.vocab_tasks is not None and task.name in args.vocab_tasks:
            vocab_sets.extend(split)

    for task in args.val_tasks:
        logger.info(f'Loading {task.name}')
        kwargs = {'test': None}
        kwargs['subsample'] = args.subsample
        kwargs['train'] = None
        kwargs['skip_cache'] = args.skip_cache
        kwargs['cached_path'] = os.path.join(args.cache, task.name)

        logger.info(f'Adding {task.name} to validation datasets')
        split = task.get_splits(args.data, lower=args.lower, **kwargs)
        assert len(split) == 1
        logger.info(f'{task.name} has {len(split[0])} validation examples')
        val_sets.append(split[0])
        if args.vocab_tasks is not None and task.name in args.vocab_tasks:
            vocab_sets.extend(split)

    numericalizer, context_embeddings, question_embeddings, decoder_embeddings = \
        load_embeddings(args.embeddings,
                        args.context_embeddings,
                        args.question_embeddings,
                        args.decoder_embeddings,
                        args.max_generative_vocab,
                        logger)
    if args.load is not None:
        numericalizer.load(args.save)
    else:
        vocab_sets = (train_sets + val_sets) if len(vocab_sets) == 0 else vocab_sets
        logger.info(f'Building vocabulary')
        numericalizer.build_vocab(Example.vocab_fields, vocab_sets)
        numericalizer.save(args.save)

    logger.info(f'Initializing encoder and decoder embeddings')
    for vec in set(context_embeddings + question_embeddings + decoder_embeddings):
        vec.init_for_vocab(numericalizer.vocab)

    logger.info(f'Vocabulary has {numericalizer.num_tokens} tokens')
    logger.debug(f'The first 200 tokens:')
    logger.debug(numericalizer.vocab.itos[:200])

    if args.use_curriculum:
        logger.info('Preprocessing auxiliary data for curriculum')
        preprocess_examples(args, args.train_tasks, aux_sets, logger, train=True)
    logger.info('Preprocessing training data')
    preprocess_examples(args, args.train_tasks, train_sets, logger, train=True)
    logger.info('Preprocessing validation data')
    preprocess_examples(args, args.val_tasks, val_sets, logger, train=args.val_filter)

    return numericalizer, context_embeddings, question_embeddings, decoder_embeddings, train_sets, val_sets, aux_sets

accumulated_batch_lengths = 0

def train_step(model, batch, iteration, opt, devices, lr_scheduler=None, grad_clip=None, pretraining=False,
               train_context_embeddings_after=None, train_question_embeddings_after=None,
               gradient_accumulation_steps=1):
    # Since the batch size is different in each call to this function due to dynamic batching, we need to keep track of
    # the total batch size
    global accumulated_batch_lengths
    model.train()
    model.module.set_train_context_embeddings(train_context_embeddings_after is not None and
                                              iteration > train_context_embeddings_after)
    model.module.set_train_question_embeddings(train_question_embeddings_after is not None and
                                               iteration > train_question_embeddings_after)
    if (iteration) % gradient_accumulation_steps == 0:
        opt.zero_grad()
    loss, predictions = model(batch, iteration, pretraining=pretraining)
    if torch.isnan(loss).any():
        raise RuntimeError('Got NaN loss')
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
            # print('p.grad = ', p.grad)
            p.grad /= accumulated_batch_lengths
        accumulated_batch_lengths = 0
        if grad_clip > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.params, grad_clip)
        opt.step()
        if lr_scheduler is not None:
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
        val_loss, metric_dict = validate(val_task, val_iter, model, logger, numericalizer,
                                         iteration, num_print=args.num_print, args=args)
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

    return best_decascore


def do_log_training_loss(iteration, loss, *,
                         lr_scheduler, grad_norm, lr_rate,
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
        else:
            writer.add_scalar(f'{log_prefix}/lr', lr_rate, iteration)
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
          start_iteration=1, rnd=1, best_decascore, use_curriculum, pretraining):
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
                                         grad_clip=args.grad_clip, pretraining=pretraining,
                                         gradient_accumulation_steps=args.gradient_accumulation_steps,
                                         train_context_embeddings_after=args.train_context_embeddings_after if
                                                                        args.train_context_embeddings else None,
                                         train_question_embeddings_after=args.train_question_embeddings_after if
                                                                         args.train_question_embeddings else None)
            if loss is None:
                logger.info(
                    'Encountered NAN loss during training... Continue training ignoring the current batch')
                continue
            if loss < 1e-5:
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
                                     lr_scheduler=lr_scheduler, grad_norm=grad_norm, lr_rate=args.lr_rate,
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


def init_model(args, numericalizer, context_embeddings, question_embeddings, decoder_embeddings, devices, logger,
               save_dict):
    model_name = args.model
    logger.info(f'Initializing {model_name}')
    Model = getattr(models, model_name)
    model = Model(numericalizer, args, context_embeddings, question_embeddings, decoder_embeddings)
    params = get_trainable_params(model)
    log_model_size(logger, model, model_name)

    if save_dict is not None:
        logger.info(f'Loading model from {os.path.join(args.save, args.load)}')
        save_dict = torch.load(os.path.join(args.save, args.load))
        model.load_state_dict(save_dict['model_state_dict'])

    model.to(devices[0])
    model = NamedTupleCompatibleDataParallel(model, device_ids=devices)
    model.params = params

    return model


def get_transformer_learning_rate(i, *, dimension, warmup):
    i += 1
    return 1. / math.sqrt(dimension) * min(1 / math.sqrt(i), i / (warmup * math.sqrt(warmup)))


def get_sgd_learning_rate(i, *, warmup):
    i += 1
    return min(math.sqrt(warmup) / math.sqrt(i), i / warmup)


def init_opt(args, model, logger):
    if args.optimizer == 'adam':
        if args.transformer_lr:
            opt = torch.optim.Adam(model.params, lr=args.transformer_lr_multiply, betas=(0.9, 0.98), eps=1e-9,
                                   weight_decay=args.weight_decay)
            lr_lambda = partial(get_transformer_learning_rate, dimension=args.dimension, warmup=args.warmup)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        else:
            opt = torch.optim.Adam(model.params, lr=args.lr_rate, betas=(args.beta0, 0.999),
                                   weight_decay=args.weight_decay)
            scheduler = None
    elif args.optimizer == 'radam':
        import radam
        if args.transformer_lr:
            logger.warning('--transformer_lr has no effect with RAdam optimizer, warmup is never applied')
        opt = radam.RAdam(model.params, lr=args.lr_rate, betas=(args.beta0, 0.999), weight_decay=args.weight_decay)
        scheduler = None
    else:
        assert args.optimizer == 'sgd'
        if args.transformer_lr:
            opt = torch.optim.SGD(model.params, lr=args.transformer_lr_multiply, weight_decay=args.weight_decay, )
            lr_lambda = partial(get_sgd_learning_rate, warmup=args.warmup)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        else:
            opt = torch.optim.SGD(model.params, lr=args.lr_rate, weight_decay=args.weight_decay, )
            scheduler = None

    return opt, scheduler


def main(args):
    args = arguments.post_parse(args)
    if args is None:
        return

    set_seed(args)
    devices = init_devices(args, args.devices)
    logger = initialize_logger(args)
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    save_dict = None
    if args.load is not None:
        logger.info(f'Loading vocab from {os.path.join(args.save, args.load)}')
        save_dict = torch.load(os.path.join(args.save, args.load))
    numericalizer, context_embeddings, question_embeddings, decoder_embeddings, train_sets, val_sets, aux_sets = \
        prepare_data(args, logger)
    if (args.use_curriculum and aux_sets is None) or (not args.use_curriculum and len(aux_sets)):
        logging.error('sth unpleasant is happening with curriculum')

    logger.info(f'Processing')
    logger.start = time.time()

    model = init_model(args, numericalizer, context_embeddings, question_embeddings, decoder_embeddings,
                       devices, logger, save_dict)
    opt, lr_scheduler = init_opt(args, model, logger)
    start_iteration = 1


    if save_dict is not None and args.resume:
        logger.info(f'Resuming Training from {os.path.splitext(args.load)[0]}_optim.pth')
        opt_state_dict = torch.load(os.path.join(args.save, f'{os.path.splitext(args.load)[0]}_optim.pth'))
        start_iteration = opt_state_dict.pop('start_iteration')
        logger.info(f'Starting iteration is {start_iteration}')
        opt.load_state_dict(opt_state_dict)

    if hasattr(args, 'tensorboard') and args.tensorboard:
        logger.info(f'Initializing Writer')
        writer = SummaryWriter(log_dir=args.tensorboard_dir, purge_step=start_iteration)
    else:
        writer = None

    if not args.resume and args.pretrain_context > 0:
        pretrain_opt, pretrain_lr_scheduler = init_opt(args, model, logger)
        train_iterations = [args.pretrain_context for _ in args.train_tasks]
        train(args, devices, model, pretrain_opt, pretrain_lr_scheduler, train_sets,
              train_iterations, numericalizer, val_sets=[], aux_sets=[], logger=logger, writer=writer,
              log_every=args.log_every, val_every=None, save_every=None, use_curriculum=False,
              rounds=len(train_sets) > 1, start_iteration=start_iteration, best_decascore=0,
              pretraining=True, log_prefix='pretrain')

    train(args, devices, model, opt, lr_scheduler, train_sets,
          args.train_iterations, numericalizer, val_sets=val_sets, aux_sets=aux_sets, logger=logger, writer=writer,
          log_every=args.log_every, val_every=args.val_every, save_every=args.save_every,
          rounds=len(train_sets) > 1, start_iteration=start_iteration, use_curriculum=args.use_curriculum,
          best_decascore=save_dict.get('best_decascore') if save_dict is not None else None,
          pretraining=False, log_prefix='training')
