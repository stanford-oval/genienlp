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
from pprint import pformat

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import arguments, models
from .arguments import save_args
from .model_utils.optimizer import init_opt
from .model_utils.parallel_utils import NamedTupleCompatibleDataParallel
from .model_utils.saver import Saver
from .ned.ned_utils import init_ned_model
from .util import (
    elapsed_time,
    get_devices,
    get_trainable_params,
    log_model_size,
    make_data_loader,
    ned_dump_entity_type_pairs,
    set_seed,
)
from .validate import print_results, validate


def initialize_logger(args):
    # set up file logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(
        os.path.join(args.log_dir, 'train.log'), maxBytes=1024 * 1024 * 10, backupCount=1
    )
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
    # initialize ned_model
    ned_model = init_ned_model(args)

    train_sets, val_sets, aux_sets = [], [], []

    train_eval_shared_kwargs = {
        'subsample': args.subsample,
        'skip_cache': args.skip_cache,
        'cache_input_data': args.cache_input_data,
        'num_workers': args.num_workers,
    }

    if any(args.train_iterations):
        for task in args.train_tasks:
            logger.info(f'Loading {task.name}')
            kwargs = {'test': None, 'validation': None}
            kwargs.update(train_eval_shared_kwargs)
            kwargs['all_dirs'] = args.train_src_languages
            kwargs['cached_path'] = os.path.join(args.cache, task.name)
            kwargs['crossner_domains'] = args.crossner_domains
            if args.use_curriculum:
                kwargs['curriculum'] = True

            logger.info(f'Adding {task.name} to training datasets')
            t0 = time.time()
            splits, paths = task.get_splits(args.data, lower=args.lower, **kwargs)

            t1 = time.time()
            logger.info('Data loading took {:.2f} seconds'.format(t1 - t0))
            assert not splits.eval and not splits.test
            if args.use_curriculum:
                assert splits.aux
                aux_sets.append(splits.aux)
                logger.info(f'{task.name} has {len(splits.aux)} auxiliary examples')
            else:
                assert splits.train

            if task.name.startswith('almond'):
                args.db_unk_id = 0
                if args.do_ned:
                    if ned_model:
                        args.num_db_types = len(ned_model.typeqid2id)
                else:
                    args.num_db_types = 0
            else:
                args.db_unk_id = 0
                args.num_db_types = 0
            save_args(args, force_overwrite=True)

            if ned_model:
                ned_model.process_examples(splits.train.examples, paths.train, task.utterance_field)

            train_sets.append(splits.train)
            logger.info(f'{task.name} has {len(splits.train)} training examples')

        for task in args.val_tasks:
            logger.info(f'Loading {task.name}')
            kwargs = {'train': None, 'test': None}
            # choose best model based on this dev set
            if args.eval_set_name is not None:
                kwargs['validation'] = args.eval_set_name
            kwargs.update(train_eval_shared_kwargs)
            kwargs['all_dirs'] = args.eval_src_languages
            kwargs['cached_path'] = os.path.join(args.cache, task.name)
            kwargs['crossner_domains'] = args.crossner_domains
            kwargs['hf_test_overfit'] = args.hf_test_overfit

            logger.info(f'Adding {task.name} to validation datasets')
            splits, paths = task.get_splits(args.data, lower=args.lower, **kwargs)

            assert not splits.train and not splits.test and not splits.aux
            logger.info(f'{task.name} has {len(splits.eval)} validation examples')

            if ned_model:
                ned_model.process_examples(splits.eval.examples, paths.eval, task.utterance_field)

            val_sets.append(splits.eval)

    if hasattr(ned_model, 'all_schema_types'):
        logger.info(f"train all_schema_types: {ned_model.all_schema_types}")

    return train_sets, val_sets, aux_sets


accumulated_batch_lengths = 0


def train_step(model, batch, iteration, opt, devices, lr_scheduler=None, grad_clip=None, gradient_accumulation_steps=1):
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
    loss = loss * len(batch[0])
    accumulated_batch_lengths += len(batch[0])

    loss.backward()
    grad_norm = None
    if (iteration + 1) % gradient_accumulation_steps == 0:
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


def do_validate(
    iteration, args, model, numericalizer, val_iters, *, train_task, round_progress, task_progress, writer, logger
):
    deca_score = 0
    for val_task_idx, (val_task, val_iter) in enumerate(val_iters):
        output, metric_dict = validate(val_task, val_iter, model, numericalizer, args, num_print=args.num_print)
        val_loss = output.loss
        if val_loss is not None:
            log_entry = f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{train_task.name}:{task_progress}val_{val_task.name}:val_loss_{val_loss:.4f}:'
            writer.add_scalar(f'loss/{val_task.name}/val', val_loss, iteration)
        else:
            log_entry = f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{train_task.name}:{task_progress}val_{val_task.name}:'

        metric_entry = ''
        for metric_key, metric_value in metric_dict.items():
            metric_entry += f'{metric_key}_{metric_value:.2f}:'
        metric_entry = metric_entry[:-1]

        deca_metric = val_task.metrics[0]
        if deca_metric == 'loss':
            deca_score += val_loss
        else:
            deca_score += metric_dict[deca_metric]

        # val log
        logger.info(log_entry + metric_entry)
        if writer is not None:
            for metric_key, metric_value in metric_dict.items():
                writer.add_scalar(f'{val_task.name}/{metric_key}/val', metric_value, iteration)
    if writer is not None:
        writer.add_scalar('deca/val', deca_score, iteration)
    logger.info(
        f'{args.timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{train_task.name}:{task_progress}val_deca:deca_{deca_score:.2f}'
    )

    return deca_score


def maybe_save(
    iteration,
    model,
    opt,
    deca_score,
    best_decascore,
    *,
    saver,
    logger,
    train_task,
    round_progress,
    task_progress,
    timestamp,
    log_dir,
    model_parallel,
):
    save_wo_finetuning = bool(best_decascore == -1)
    should_save_best = False
    if deca_score is not None and (best_decascore is None or best_decascore < deca_score):
        best_decascore = deca_score
        should_save_best = True

    # DataParallel and ModelParallel are mutually exclusive
    if model_parallel:
        model_state_dict = model.state_dict()
    else:
        # punch through the nn.DataParallel to access the real model, otherwise we won't be able
        # to load this model later
        model_state_dict = model.module.state_dict()

    model_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}

    save_model_state_dict = {'model_state_dict': model_state_dict, 'best_decascore': best_decascore}
    save_opt_state_dict = opt.state_dict()
    save_opt_state_dict.update({'start_iteration': iteration})

    if not save_wo_finetuning:
        saver.save(save_model_state_dict, save_opt_state_dict, global_step=iteration)
    if should_save_best:
        logger.info(
            f'{timestamp}:{elapsed_time(logger)}:iteration_{iteration}:{round_progress}train_{train_task.name}:{task_progress} saving new best model'
        )
        torch.save(save_model_state_dict, os.path.join(log_dir, 'best.pth'))
        if not save_wo_finetuning:
            torch.save(save_opt_state_dict, os.path.join(log_dir, 'best_optim.pth'))

        if model_parallel:
            model.numericalizer.save(saver._savedir)
        else:
            model.module.numericalizer.save(saver._savedir)

    return best_decascore


def do_log_training_loss(
    iteration,
    loss,
    *,
    lr_scheduler,
    grad_norm,
    num_examples,
    len_contexts,
    len_answers,
    logger,
    train_task,
    round_progress,
    epochs,
    task_progress,
    timestamp,
    writer,
    log_prefix,
):
    avg_batch_size = f'avbatch_{num_examples:.0f}_{len_contexts:.0f}_{len_answers:.0f}:'
    logger.info(
        f'{timestamp}:{elapsed_time(logger)}:iteration_{iteration}:epoch_{epochs:.2f}:{round_progress}train_{train_task.name}:{task_progress}{avg_batch_size}{log_prefix}/loss_{loss:.4f}'
    )

    if writer is not None:
        writer.add_scalar(f'{log_prefix}/loss/{train_task.name}', loss, iteration)

        if lr_scheduler is not None:
            writer.add_scalar(f'{log_prefix}/lr', np.array(lr_scheduler.get_last_lr()), iteration)
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


def train(
    args,
    devices,
    model,
    opt,
    lr_scheduler,
    train_sets,
    train_iterations,
    numericalizer,
    *,
    log_every,
    val_every,
    save_every,
    rounds,
    val_sets,
    aux_sets,
    writer,
    logger,
    log_prefix,
    start_iteration=1,
    rnd=1,
    best_decascore,
    use_curriculum,
):
    """main training function"""
    local_loss, num_examples, len_contexts, len_answers, iteration = 0, 0, 0, 0, 1

    train_iter_deep = deepcopy(train_iterations)

    task_iteration = dict()
    task_done = dict()
    task_fraction = dict()
    task_total_num_examples = dict()
    task_train_size = dict()  # number of examples in each task

    for task, train_set in zip(args.train_tasks, train_sets):
        task_iteration[task] = 1
        task_done[task] = False
        task_fraction[task] = 0.0
        task_total_num_examples[task] = 0.0
        task_train_size[task] = len(train_set)

    saver = Saver(args.log_dir, args.max_to_keep)
    per_task_iterations = 0

    logger.info('Preparing iterators')
    main_device = devices[0]

    t0 = time.time()
    train_iters = [
        (task, make_data_loader(dataset, numericalizer, tok, main_device, train=True))
        for task, dataset, tok in zip(args.train_tasks, train_sets, args.train_batch_tokens)
    ]
    t1 = time.time()
    logger.info('Preparing iterators took {:.2f} seconds'.format(t1 - t0))

    train_iters = [(task, iter(train_iter)) for task, train_iter in train_iters]
    # save memory
    del train_sets

    val_iters = [
        (task, make_data_loader(dataset, numericalizer, bs, main_device, train=False))
        for task, dataset, bs in zip(args.val_tasks, val_sets, args.val_batch_size)
    ]
    # save memory
    del val_sets

    aux_iters = []
    if use_curriculum:
        aux_iters = [
            (name, make_data_loader(dataset, numericalizer, tok, main_device, train=True))
            for name, dataset, tok in zip(args.train_tasks, aux_sets, args.train_batch_tokens)
        ]
        aux_iters = [(task, iter(aux_iter)) for task, aux_iter in aux_iters]
        # save memory
        del aux_sets

    zero_loss = 0
    logger.info(f'Begin {log_prefix}')

    if any(train_iterations):
        while not all(task_done.values()):
            # For some number of rounds, we 'jump start' some subset of the tasks
            # by training them and not others
            # once the specified number of rounds is completed,
            # switch to normal round robin training
            if rnd < args.jump_start:
                train_iterations = [0] * len(train_iterations)
                for j in range(args.n_jump_start):
                    train_iterations[j] = 1
            else:
                train_iterations = train_iter_deep

            for task_idx, (task, train_iter) in enumerate(train_iters):
                task_iterations = train_iterations[task_idx] if train_iterations is not None else None
                if task_iterations == 0:
                    continue
                if task_iterations is not None and task_iteration[task] > task_iterations:
                    task_done[task] = True
                    continue

                # load batches even if (args.resume == True) and we are going to skip the iteration
                # this makes runs that are resumed have the exact same behavior as runs that are
                # finished in one pass (given that the random seed is the same).
                batch = get_next_batch(
                    train_iter,
                    aux_iters,
                    task=task,
                    task_idx=task_idx,
                    task_fraction=task_fraction,
                    use_curriculum=use_curriculum,
                )

                if iteration < start_iteration:
                    # skip this iteration (this is done to ensure iterators are at the same position when resuming)
                    task_iteration[task] += 1
                    iteration += 1
                    if (iteration + 1) % args.gradient_accumulation_steps == 0:
                        lr_scheduler.step()  # update the learning rate
                    continue

                task_progress = f'{task_iteration[task]}/{task_iterations}:' if task_iterations is not None else ''
                round_progress = f'round_{rnd}:' if rounds else ''

                # param update
                loss, grad_norm = train_step(
                    model,
                    batch,
                    iteration,
                    opt,
                    devices,
                    lr_scheduler=lr_scheduler,
                    grad_clip=args.grad_clip,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                )
                if loss is None:
                    logger.info('Encountered NAN loss during training. Continue training ignoring the current batch')
                    continue
                if loss < 1e-6:
                    zero_loss += 1
                    if zero_loss >= 100:
                        logger.info('Found loss less than 1e-6 for 100 steps, stopping.')
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

                task_total_num_examples[task] += batch.context.value.size(0)

                if should_log(iteration, log_every):
                    local_loss /= log_every
                    num_examples /= log_every
                    len_contexts /= log_every
                    len_answers /= log_every
                    do_log_training_loss(
                        iteration,
                        local_loss,
                        lr_scheduler=lr_scheduler,
                        grad_norm=grad_norm,
                        num_examples=num_examples,
                        len_contexts=len_contexts,
                        len_answers=len_answers,
                        logger=logger,
                        writer=writer,
                        train_task=task,
                        round_progress=round_progress,
                        epochs=task_total_num_examples[task] / task_train_size[task],
                        task_progress=task_progress,
                        timestamp=args.timestamp,
                        log_prefix=log_prefix,
                    )
                    num_examples = 0
                    len_contexts = 0
                    len_answers = 0
                    local_loss = 0

                # validate
                if should_validate(iteration, val_every, resume=args.resume, start_iteration=start_iteration):
                    if args.print_train_examples_too:
                        results = {
                            'answer': numericalizer.reverse(batch.answer.value.data, 'answer'),
                            'context': numericalizer.reverse(batch.context.value.data, 'context'),
                        }
                        num_print = min(len(results['answer']), args.num_print)
                        print_results(results, num_print)

                    deca_score = do_validate(
                        iteration,
                        args,
                        model,
                        numericalizer,
                        val_iters,
                        train_task=task,
                        round_progress=round_progress,
                        task_progress=task_progress,
                        writer=writer,
                        logger=logger,
                    )

                    # saving
                    if should_save(iteration, save_every):
                        best_decascore = maybe_save(
                            iteration,
                            model,
                            opt,
                            deca_score,
                            best_decascore,
                            saver=saver,
                            logger=logger,
                            train_task=task,
                            round_progress=round_progress,
                            task_progress=task_progress,
                            timestamp=args.timestamp,
                            log_dir=args.log_dir,
                            model_parallel=args.model_parallel,
                        )

                # book keeping
                task_iteration[task] += 1
                iteration += 1

            # book keeping
            per_task_iterations += 1
            rnd += 1

        logger.info(f'{log_prefix} is done after {per_task_iterations - 1} iterations')

    else:
        # Save pretrained models as is without any finetuning
        # Useful for doing prediction/ generation on those models with genienlp
        for task in args.train_tasks:
            maybe_save(
                0,
                model,
                opt,
                deca_score=0,
                best_decascore=-1,
                saver=saver,
                logger=logger,
                train_task=task,
                round_progress=0,
                task_progress=0,
                timestamp=args.timestamp,
                log_dir=args.log_dir,
                model_parallel=args.model_parallel,
            )

        logger.info(f'{args.pretrained_model} model is saved to {args.save} without any fine-tuning')


def main(args):
    args = arguments.post_parse_general(args)
    args = arguments.post_parse_train_specific(args)
    if args is None:
        return

    set_seed(args)
    devices = get_devices(args.devices)
    logger = initialize_logger(args)
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    model_name = args.model
    model_class = getattr(models, model_name)

    tasks = set(args.train_tasks) | set(args.val_tasks)
    train_sets, val_sets, aux_sets = prepare_data(args, logger)

    if (args.use_curriculum and aux_sets is None) or (not args.use_curriculum and len(aux_sets) > 0):
        logging.error('Something unpleasant is happening with curriculum')

    logger.info('Processing')
    logger.start = time.time()

    # TODO handle multiple languages
    # TODO handle different train and eval languages
    src_lang = args.train_src_languages.split('+')[0]
    tgt_lang = args.train_tgt_languages.split('+')[0]

    ########## initialize model
    best_decascore = None
    if args.load is not None:
        model, best_decascore = model_class.load(
            args.save,
            args=args,
            model_checkpoint_file=args.load,
            vocab_sets=train_sets + val_sets,
            tasks=tasks,
            device=devices[0],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        model.add_new_vocab_from_data(tasks=tasks, resize_decoder=True)
        if not args.resume:
            # we are fine-tuning, so reset the best score since the new fine-tune dataset usually has a different validation set from the original
            best_decascore = None
    else:
        logger.info(f'Initializing a new {model_name}')
        model = model_class(args=args, vocab_sets=train_sets + val_sets, tasks=tasks, src_lang=src_lang, tgt_lang=tgt_lang)

    # dump entities if required
    if args.ned_dump_entity_type_pairs and args.add_entities_to_text == 'append':
        for task, train_set, val_set in zip(tasks, train_sets, val_sets):
            ned_dump_entity_type_pairs(train_set, args.data, 'train', task.utterance_field)
            ned_dump_entity_type_pairs(val_set, args.data, 'eval', task.utterance_field)

    params = get_trainable_params(model)
    log_model_size(logger, model, model_name)

    if args.model_parallel:
        n_layers = len(model.model.encoder.block)
        layers = list(range(n_layers))

        if args.mp_device_ratio is not None:
            if len(args.devices) > torch.cuda.device_count() or max(args.devices) >= torch.cuda.device_count():
                raise ValueError('Provided GPU devices exceeds the number of available GPU')

            mp_device_ratio = [(device_ratio / sum(args.mp_device_ratio)) for device_ratio in args.mp_device_ratio]
            layers_list = []
            beg = 0
            for i in range(len(args.devices)):
                end = min(beg + math.ceil(n_layers * mp_device_ratio[i]), n_layers)
                layers_list.append(layers[beg:end])
                beg = end

            if any([len(val_list) == 0 for val_list in layers_list]):
                raise ValueError(
                    'One device got no layers given the mp_device_ratio you provided'
                    'Hint: if you do not provide mp_device_ratio, layers will be distributed evenly'
                )

        else:
            n_blocks = int(math.ceil(n_layers / len(args.devices)))
            layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

        device_map = dict(zip(args.devices, layers_list))
        model.model.parallelize(device_map)
        logger.info(f'Model parallel is used with following device map: {model.model.device_map}')
    else:
        model.to(devices[0])
        model = NamedTupleCompatibleDataParallel(model, device_ids=devices)
    model.params = params
    ##########

    opt, lr_scheduler = init_opt(args, model, logger)
    start_iteration = 1

    if args.resume:
        logger.info(f'Resuming training from {os.path.splitext(args.load)[0]}_optim.pth')
        # load optimizer's state_dict to cpu first to avoid GPU memory surge. Will crash with OOM if `map_location='cpu'` is not specified.
        opt_state_dict = torch.load(os.path.join(args.save, f'{os.path.splitext(args.load)[0]}_optim.pth'), map_location='cpu')
        start_iteration = opt_state_dict.pop('start_iteration')
        logger.info(f'Starting iteration is {start_iteration}')
        opt.load_state_dict(opt_state_dict)

    if hasattr(args, 'tensorboard') and args.tensorboard:
        logger.info('Initializing Writer')
        writer = SummaryWriter(log_dir=args.tensorboard_dir, purge_step=start_iteration, flush_secs=60)
    else:
        writer = None

    train(
        args,
        devices,
        model,
        opt,
        lr_scheduler,
        train_sets,
        args.train_iterations,
        model.module.numericalizer if not args.model_parallel else model.numericalizer,
        val_sets=val_sets,
        aux_sets=aux_sets,
        logger=logger,
        writer=writer,
        log_every=args.log_every,
        val_every=args.val_every,
        save_every=args.save_every,
        rounds=len(train_sets) > 1,
        start_iteration=start_iteration,
        use_curriculum=args.use_curriculum,
        best_decascore=best_decascore,
        log_prefix='training',
    )

    if writer is not None:
        writer.close()  # otherwise the last written value may not be flushed
