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


import os
import math
import time
import sys
from copy import deepcopy

import logging
from pprint import pformat
from logging import handlers
from collections import defaultdict
import numpy as np
import itertools

import torch

from .text import torchtext

from tensorboardX import SummaryWriter

from . import arguments
from . import models
from .validate import validate
from .multiprocess import Multiprocess, DistributedDataParallel
from .util import elapsed_time, batch_fn, set_seed, preprocess_examples, get_trainable_params, count_params
from .utils.saver import Saver
from .utils.embeddings import load_embeddings


def initialize_logger(args, rank='main'):
    # set up file logger
    logger = logging.getLogger(f'process_{rank}')
    logger.setLevel(logging.DEBUG)
    handler = handlers.RotatingFileHandler(os.path.join(args.log_dir, f'process_{rank}.log'), maxBytes=1024*1024*10, backupCount=1)
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


def log(rank='main'):
    return logging.getLogger(f'process_{rank}')


def prepare_data(args, field, logger):

    if field is None:
        logger.info(f'Constructing field')
        FIELD = torchtext.data.ReversibleField(batch_first=True, init_token='<init>', eos_token='<eos>', lower=args.lower, include_lengths=True)
    else:
        FIELD = field

    train_sets, val_sets, aux_sets, vocab_sets = [], [], [], []
    for task in args.train_tasks:
        logger.info(f'Loading {task.name}')
        kwargs = {'test': None}
        kwargs['subsample'] = args.subsample
        kwargs['validation'] = None
        if args.use_curriculum:
            kwargs['curriculum'] = True
        kwargs['skip_cache_bool'] = args.skip_cache_bool
        kwargs['cached_path'] = args.cached

        logger.info(f'Adding {task.name} to training datasets')
        split = task.get_splits(FIELD, args.data, **kwargs)
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
        kwargs['skip_cache_bool'] = args.skip_cache_bool
        kwargs['cached_path'] = args.cached

        logger.info(f'Adding {task.name} to validation datasets')
        split = task.get_splits(FIELD, args.data, **kwargs)
        assert len(split) == 1
        logger.info(f'{task.name} has {len(split[0])} validation examples')
        val_sets.append(split[0])
        if args.vocab_tasks is not None and task.name in args.vocab_tasks:
            vocab_sets.extend(split)

    if args.load is None:
        vectors = load_embeddings(args, logger)
        vocab_sets = (train_sets + val_sets) if len(vocab_sets) == 0 else vocab_sets
        logger.info(f'Building vocabulary')
        FIELD.build_vocab(*vocab_sets, max_size=args.max_effective_vocab, vectors=vectors)

    FIELD.decoder_itos = FIELD.vocab.itos[:args.max_generative_vocab]
    FIELD.decoder_stoi = {word: idx for idx, word in enumerate(FIELD.decoder_itos)}
    FIELD.decoder_to_vocab = {idx: FIELD.vocab.stoi[word] for idx, word in enumerate(FIELD.decoder_itos)}
    FIELD.vocab_to_decoder = {idx: FIELD.decoder_stoi[word] for idx, word in enumerate(FIELD.vocab.itos) if word in FIELD.decoder_stoi}

    logger.info(f'Vocabulary has {len(FIELD.vocab)} tokens')
    logger.debug(f'The first 200 tokens:')
    logger.debug(FIELD.vocab.itos[:200])

    if args.use_curriculum:
        logger.info('Preprocessing auxiliary data for curriculum')
        preprocess_examples(args, args.train_tasks, aux_sets, FIELD, logger, train=True)
    logger.info('Preprocessing training data')
    preprocess_examples(args, args.train_tasks, train_sets, FIELD, logger, train=True)
    logger.info('Preprocessing validation data')
    preprocess_examples(args, args.val_tasks, val_sets, FIELD, logger, train=args.val_filter)

    return FIELD, train_sets, val_sets, aux_sets


def to_iter(args, world_size, val_batch_size, data, device, train=True, token_testing=False, sort=None):
    sort = sort if not token_testing else True
    shuffle = None if not token_testing else False
    reverse = args.reverse
    Iterator = torchtext.data.BucketIterator if train else torchtext.data.Iterator
    it = Iterator(data, batch_size=val_batch_size,
       device=device, batch_size_fn=batch_fn if train else None, 
       distributed=world_size>1, train=train, repeat=train, sort=sort,
       shuffle=shuffle, reverse=reverse)
    return it


def get_learning_rate(i, args):
    transformer_lr = 1. / math.sqrt(args.dimension) * min(
        1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
    if 'adam' not in args.optimizer.lower():
        transformer_lr = transformer_lr * math.sqrt(args.dimension * args.warmup) * args.sgd_lr
    return transformer_lr


def step(model, batch, opt, iteration, field, task, lr=None, grad_clip=None, writer=None, it=None):
    model.train()
    opt.zero_grad()
    loss, predictions = model(batch, iteration)
    loss.backward()
    if lr is not None:
        opt.param_groups[0]['lr'] = lr
    grad_norm = None
    if grad_clip > 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.params, grad_clip)
    opt.step()
    if torch.isnan(loss).item():
        raise ValueError('Found NaN loss')
    return loss.item(), {}, grad_norm


def update_fraction(args, task_iteration):

    if args.curriculum_strategy == 'linear':
        next_fraction = args.curriculum_rate * task_iteration
    elif args. curriculum_strategy == 'exp':
        next_fraction = args.curriculum_rate * np.exp(task_iteration)

    fraction = min(args.curriculum_max_frac, next_fraction)

    return fraction

def train(args, model, opt, train_sets, train_iterations, field, rank=0, world_size=1,
    log_every=10, val_every=100, save_every=1000, rounds=False, val_sets=[], aux_sets=[], writer=None, start_iteration=1, rnd=1, best_decascore=None):
    """main training function"""

    device = next(model.parameters()).device

    logger = log(rank) 
    local_loss, num_examples, len_contexts, len_answers, iteration = 0, 0, 0, 0, start_iteration

    train_iter_deep = deepcopy(train_iterations)
    local_train_metric_dict = dict()

    task_iteration = dict()
    task_done = dict()
    task_fraction = dict()

    for task in args.train_tasks:
        task_iteration[task] = 1
        task_done[task] = False
        task_fraction[task] = 0.0

    saver = Saver(args.log_dir, world_size, args.max_to_keep)
    epoch = 0

    logger.info(f'Preparing iterators')
    train_iters = [(task, to_iter(args, world_size, tok, x, device, token_testing=args.token_testing))
                      for task, x, tok in zip(args.train_tasks, train_sets, args.train_batch_tokens)]
    train_iters = [(task, iter(train_iter)) for task, train_iter in train_iters]

    val_iters = [(task, to_iter(args, world_size, tok, x, device, train=False, token_testing=args.token_testing, sort=False if 'sql' in task.name else None))
                    for task, x, tok in zip(args.val_tasks, val_sets, args.val_batch_size)]


    if args.use_curriculum:
        aux_iters = [(name, to_iter(args, world_size, tok, x, device, token_testing=args.token_testing))
                          for name, x, tok in zip(args.train_tasks, aux_sets, args.train_batch_tokens)]
        aux_iters = [(task, iter(aux_iter)) for task, aux_iter in aux_iters]

    while True:

        # For some number of rounds, we 'jump start' some subset of the tasks
        # by training them and not others
        # once the specified number of rounds is completed, 
        # switch to normal round robin training
        if rnd < args.jump_start:
            train_iterations = [0]*len(train_iterations)
            for j in range(args.n_jump_start): train_iterations[j] = 1
        else:
            train_iterations = train_iter_deep

        for task_idx, (task, train_iter)in enumerate(train_iters):

            task_iterations = train_iterations[task_idx] if train_iterations is not None else None
            if task_iterations == 0:
                continue
            if task_iterations is not None and task_iteration[task] > task_iterations:
                task_done[task] = True
                continue

            if args.use_curriculum:
                aux_iter = aux_iters[task_idx][1]
                prob = np.random.choice(['train', 'aux'], p=[1-task_fraction[task], task_fraction[task]])
                if prob == 'aux':
                    batch = next(aux_iter)
                elif prob == 'train':
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
                            val_loss, metric_dict = validate(val_task, val_iter, model, logger, field, world_size, rank, iteration, num_print=args.num_print, args=args)
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
                                    writer.add_scalar(f'{metric_key}/{val_task.name}/val', metric_value, iteration)
                                    writer.add_scalar(f'{val_task.name}/{metric_key}/val', metric_value, iteration)
                        writer.add_scalar('deca/val', deca_score, iteration)
                        logger.info(f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task.name}:{task_progress}val_deca:deca_{deca_score:.2f}')

                    # saving
                    if save_every is not None and (iteration % args.save_every == 0):
                        if rank is not None and rank == 0:
                            should_save_best = False
                            if deca_score is not None and (best_decascore is None or best_decascore < deca_score):
                                best_decascore = deca_score
                                should_save_best = True
                                
                            save_model_state_dict = {'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'field': field,
                                               'best_decascore': best_decascore}
                            save_opt_state_dict = opt.state_dict()
                            save_opt_state_dict.update({'start_iteration': iteration})

                            if world_size > 1:
                                torch.distributed.barrier()
                            saver.save(save_model_state_dict, save_opt_state_dict, global_step=iteration)
                            if should_save_best:
                                logger.info(f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task.name}:{task_progress}found new best model')
                                torch.save(save_model_state_dict, os.path.join(args.log_dir, 'best.pth'))
                                if world_size > 1:
                                    torch.distributed.barrier()
                                torch.save(save_opt_state_dict, os.path.join(args.log_dir, 'best_optim.pth'))
                                if world_size > 1:
                                    torch.distributed.barrier()

                    # lr update
                    lr = opt.param_groups[0]['lr'] 
                    if args.warmup > 0 and args.transformer_lr:
                        lr = get_learning_rate(iteration, args) 

                    # param update
                    loss, train_metric_dict, grad_norm = step(model, batch, opt, iteration, field, task, lr=lr, grad_clip=args.grad_clip, writer=writer, it=train_iter)

                    # update curriculum fraction
                    if args.use_curriculum:
                        task_fraction[task] = update_fraction(args, task_iteration[task])

                    # train metrics
                    local_loss += loss
                    for metric_name, metric_val in train_metric_dict.items():
                        if metric_name in local_train_metric_dict:
                            local_train_metric_dict[metric_name] += metric_val / args.log_every
                        else:
                            local_train_metric_dict[metric_name] = metric_val / args.log_every

                    # train logs
                    num_examples += batch.context.size(0)
                    len_contexts += batch.context.size(1)
                    len_answers += batch.answer.size(1)

                    if log_every is not None and (iteration % log_every == 0 % log_every):
                        local_loss /= args.log_every
                        num_examples /= args.log_every
                        len_contexts /= args.log_every
                        len_answers /= args.log_every
                        avg_batch_size = f'avbatch_{num_examples:.0f}_{len_contexts:.0f}_{len_answers:.0f}:'
                        metric_entry = ''
                        for metric_key, metric_value in local_train_metric_dict.items():
                            metric_entry += f'{metric_key}_{metric_value:.2f}:'
                        metric_entry = f'{metric_entry[:-1]}'
                        logger.info(f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{task.name}:{task_progress}{avg_batch_size}loss_{local_loss:.4f}{metric_entry}')
                        num_examples = 0 
                        len_contexts = 0 
                        len_answers = 0  
    
                        if writer is not None:
                            writer.add_scalar(f'loss/{task.name}/train', local_loss, iteration)
                            writer.add_scalar(f'training/lr', lr, iteration)
                            if grad_norm is not None:
                                writer.add_scalar(f'training/norm', grad_norm, iteration)
                            for metric_key, metric_value in local_train_metric_dict.items():
                                writer.add_scalar(f'{metric_key}/{task.name}/train', metric_value, iteration)
                                writer.add_scalar(f'{task.name}/{metric_key}/train', metric_value, iteration)

                        local_loss = 0
                        local_train_metric_dict = {}
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


def run(args, run_args, rank=0, world_size=1):
    device = set_seed(args, rank=rank)
    logger = initialize_logger(args, rank)
    field, train_sets, val_sets, aux_sets, save_dict = run_args

    logger.start = time.time()

    if hasattr(args, 'tensorboard') and args.tensorboard:
        logger.info(f'Initializing Writer')
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    model = init_model(args, field, logger, world_size, device)
    opt = init_opt(args, model) 
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

    logger.info(f'Begin Training')
    train(args, model, opt, train_sets, args.train_iterations, field, val_sets=val_sets, aux_sets=aux_sets,
        rank=rank, world_size=world_size, 
        log_every=args.log_every, val_every=args.val_every, rounds=len(train_sets)>1,
        writer=writer if rank==0 else None, save_every=args.save_every, start_iteration=start_iteration,
        best_decascore=save_dict.get('best_decascore') if save_dict is not None else None)


def init_model(args, field, logger, world_size, device):
    logger.info(f'Initializing {args.model}')
    Model = getattr(models, args.model) 
    model = Model(field, args)
    params = get_trainable_params(model) 
    num_param = count_params(params)
    logger.info(f'{args.model} has {num_param:,} trainable parameters')

    model.to(device)
    if world_size > 1: 
        logger.info(f'Wrapping model for distributed')
        model = DistributedDataParallel(model)

    model.params = params
    return model


def init_opt(args, model):
    opt = None
    if 'adam' in args.optimizer.lower():
        if args.transformer_lr:
            opt = torch.optim.Adam(model.params, lr=args.lr_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
        else:
            opt = torch.optim.Adam(model.params, lr=args.lr_rate, betas=(args.beta0, 0.999), weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(model.params, lr=args.sgd_lr, weight_decay=args.weight_decay,)
    return opt


def main(argv=sys.argv):
    args = arguments.parse(argv)
    if args is None:
        return
    set_seed(args)
    logger = initialize_logger(args)
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    field, save_dict = None, None
    if args.load is not None:
        logger.info(f'Loading field from {os.path.join(args.save, args.load)}')
        save_dict = torch.load(os.path.join(args.save, args.load))
        field = save_dict['field']
    field, train_sets, val_sets, aux_sets = prepare_data(args, field, logger)
    if (args.use_curriculum and aux_sets is None) or (not args.use_curriculum and len(aux_sets)):
        logging.error('sth unpleasant is happening with curriculum')
    run_args = (field, train_sets, val_sets, aux_sets, save_dict)
    if len(args.devices) > 1:
        logger.info(f'Multiprocessing')
        mp = Multiprocess(run, args)
        mp.run(run_args)
    else:
        logger.info(f'Processing')
        run(args, run_args, world_size=args.world_size)


if __name__ == '__main__':
    main()

