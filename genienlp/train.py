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
from .data.embeddings import load_embeddings
from .data.example import Example
from .util import elapsed_time, set_seed, preprocess_examples, get_trainable_params, make_data_loader, log_model_size, \
    init_devices
from .utils.parallel_utils import NamedTupleCompatibleDataParallel
from .utils.saver import Saver
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

    numericalizer, encoder_embeddings, decoder_embeddings = load_embeddings(args.embeddings, args.encoder_embeddings,
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

        for vec in set(encoder_embeddings + decoder_embeddings):
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

    return numericalizer, encoder_embeddings, decoder_embeddings, train_sets, val_sets, aux_sets


def step(model, batch, iteration, opt, lr_scheduler=None, grad_clip=None, logger=None):
    model.train()
    opt.zero_grad()
    loss, predictions = model(batch, iteration)
    if torch.isnan(loss).any():
        raise RuntimeError('Got NaN loss')
    loss.backward()
    grad_norm = None
    if grad_clip > 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.params, grad_clip)
    opt.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss.item(), grad_norm


def update_fraction(args, task_iteration):
    if args.curriculum_strategy == 'linear':
        next_fraction = args.curriculum_rate * task_iteration
    elif args.curriculum_strategy == 'exp':
        next_fraction = args.curriculum_rate * np.exp(task_iteration)

    fraction = min(args.curriculum_max_frac, next_fraction)

    return fraction


def train(args, devices, model, opt, lr_scheduler, train_sets, train_iterations, numericalizer, logger,
          log_every=10, val_every=100, save_every=1000, rounds=False, val_sets=[], aux_sets=[], writer=None,
          start_iteration=1, rnd=1, best_decascore=None):
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

    if args.use_curriculum:
        aux_iters = [(name, make_data_loader(x, numericalizer, tok, main_device, train=True))
                     for name, x, tok in zip(args.train_tasks, aux_sets, args.train_batch_tokens)]
        aux_iters = [(task, iter(aux_iter)) for task, aux_iter in aux_iters]

    zero_loss = 0
    logger.info(f'Begin Training')

    while True:

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

            if args.use_curriculum:
                aux_iter = aux_iters[task_idx][1]
                prob = np.random.choice(['train', 'aux'], p=[1 - task_fraction[task], task_fraction[task]])
                if prob == 'aux':
                    batch = next(aux_iter)
                else:
                    assert prob == 'train'
                    batch = next(train_iter)

            else:
                batch = next(train_iter)

            # run only once
            for _ in range(1):
                if not args.resume or iteration > start_iteration:
                    task_progress = f'{task_iteration[task]}/{task_iterations}:' if task_iterations is not None else ''
                    round_progress = f'round_{rnd}:' if rounds else ''

                    # validate
                    deca_score = None
                    if (val_every is not None and
                            ((iteration % args.val_every == 0 % args.val_every) or
                             (args.load and iteration == start_iteration + 1))):

                        deca_score = 0
                        for val_task_idx, (val_task, val_iter) in enumerate(val_iters):
                            val_loss, metric_dict = validate(val_task, val_iter, model, logger, numericalizer,
                                                             iteration, num_print=args.num_print, args=args)
                            if val_loss is not None:
                                log_entry = f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task.name}:{task_progress}val_{val_task.name}:val_loss{val_loss.item():.4f}:'
                                writer.add_scalar(f'loss/{val_task.name}/val', val_loss.item(), iteration)
                            else:
                                log_entry = f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task.name}:{task_progress}val_{val_task.name}:'

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
                            f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task.name}:{task_progress}val_deca:deca_{deca_score:.2f}')

                    # saving
                    if save_every is not None and (iteration % args.save_every == 0):
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
                                f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task.name}:{task_progress}found new best model')
                            torch.save(save_model_state_dict, os.path.join(args.log_dir, 'best.pth'))
                            torch.save(save_opt_state_dict, os.path.join(args.log_dir, 'best_optim.pth'))

                    # param update
                    loss, grad_norm = step(model, batch, iteration, opt, lr_scheduler=lr_scheduler,
                                           grad_clip=args.grad_clip, logger=logger)
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

                    if log_every is not None and (iteration % log_every == 0 % log_every):
                        local_loss /= args.log_every
                        num_examples /= args.log_every
                        len_contexts /= args.log_every
                        len_answers /= args.log_every
                        avg_batch_size = f'avbatch_{num_examples:.0f}_{len_contexts:.0f}_{len_answers:.0f}:'
                        logger.info(
                            f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task.name}:{task_progress}{avg_batch_size}loss_{local_loss:.4f}')
                        num_examples = 0
                        len_contexts = 0
                        len_answers = 0

                        if writer is not None:
                            writer.add_scalar(f'loss/{task.name}/train', local_loss, iteration)
                            writer.add_scalar(f'training/loss/{task.name}', local_loss, iteration)

                            if lr_scheduler is not None:
                                writer.add_scalar(f'training/lr', lr_scheduler.get_last_lr(), iteration)
                            else:
                                writer.add_scalar(f'training/lr', args.lr_rate)
                            if grad_norm is not None:
                                writer.add_scalar(f'training/norm', grad_norm, iteration)

                        local_loss = 0
                        num_examples = 0

                # book keeping
                task_iteration[task] += 1
                iteration += 1

        # book keeping
        epoch += 1
        rnd += 1

        if all(task_done.values()):
            logger.info(f'training is done after {epoch} epochs')
            break


def init_model(args, numericalizer, encoder_embeddings, decoder_embeddings, devices, logger):
    model_name = args.model
    logger.info(f'Initializing {model_name}')
    Model = getattr(models, model_name)
    model = Model(numericalizer, args, encoder_embeddings, decoder_embeddings)
    params = get_trainable_params(model)
    log_model_size(logger, model, model_name)

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
    numericalizer, encoder_embeddings, decoder_embeddings, train_sets, val_sets, aux_sets = prepare_data(args, logger)
    if (args.use_curriculum and aux_sets is None) or (not args.use_curriculum and len(aux_sets)):
        logging.error('sth unpleasant is happening with curriculum')

    logger.info(f'Processing')
    logger.start = time.time()

    if hasattr(args, 'tensorboard') and args.tensorboard:
        logger.info(f'Initializing Writer')
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
    else:
        writer = None

    model = init_model(args, numericalizer, encoder_embeddings, decoder_embeddings, devices, logger)
    opt, lr_scheduler = init_opt(args, model, logger)
    start_iteration = 1

    if save_dict is not None:
        logger.info(f'Loading model from {os.path.join(args.save, args.load)}')
        save_dict = torch.load(os.path.join(args.save, args.load))
        model.load_state_dict(save_dict['model_state_dict'])
        if args.resume:
            logger.info(f'Resuming Training from {os.path.splitext(args.load)[0]}_optim.pth')
            opt_state_dict = torch.load(os.path.join(args.save, f'{os.path.splitext(args.load)[0]}_optim.pth'))
            start_iteration = opt_state_dict.pop('start_iteration')
            logger.info(f'Starting iteration is {start_iteration}')
            opt.load_state_dict(opt_state_dict)

    train(args, devices, model, opt, lr_scheduler, train_sets, args.train_iterations, numericalizer, val_sets=val_sets,
          aux_sets=aux_sets,
          logger=logger,
          log_every=args.log_every, val_every=args.val_every, rounds=len(train_sets) > 1,
          writer=writer, save_every=args.save_every, start_iteration=start_iteration,
          best_decascore=save_dict.get('best_decascore') if save_dict is not None else None)
