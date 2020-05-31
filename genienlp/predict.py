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

import json
import logging
import os
from pprint import pformat
from tqdm import tqdm
from collections import defaultdict

import torch

from . import models
from .data_utils.embeddings import load_embeddings
from .metrics import compute_metrics
from .tasks.registry import get_tasks
from .util import set_seed, preprocess_examples, load_config_json, make_data_loader, log_model_size, init_devices, \
    have_multilingual

logger = logging.getLogger(__name__)


def get_all_splits(args):
    splits = []
    if len(args.pred_languages) == 1 and len(args.tasks) > 1:
        args.pred_languages *= len(args.tasks)
    for i, task in enumerate(args.tasks):
        task_languages = args.pred_languages[i]
        logger.info(f'Loading {task}')
        kwargs = {'train': None}
        if args.evaluate == 'valid':
            kwargs['test'] = None
            if args.pred_set_name is not None:
                kwargs['validation'] = args.pred_set_name
        elif args.evaluate == 'test':
            kwargs['validation'] = None
        else:
            raise ValueError('Split used for prediction should be either valid or test')
        
        kwargs.update({'skip_cache': args.skip_cache, 'subsample': args.subsample,
                       'cached_path': os.path.join(args.cache, task.name), 'all_dirs': task_languages})
        
        kwargs['separate_eval'] = args.separate_eval
        task_splits = task.get_splits(root=args.data, lower=args.lower, **kwargs)
        if not isinstance(task_splits, list):
            task_splits = [task_splits]
        task_split_processed = []
        for split in task_splits:
            assert (split.eval or split.test) and not split.train and not split.aux
            split = split.eval if split.eval else split.test
            preprocess_examples(args, [task], [split], train=False)
            task_split_processed.append(split)
        splits.append(task_split_processed)
    return splits


def prepare_data(args, numericalizer, embeddings):
    splits = get_all_splits(args)
    logger.info(f'Vocabulary has {numericalizer.num_tokens} tokens from training')
    new_words = []
    for task_splits in splits:
        for split in task_splits:
            new_words += numericalizer.grow_vocab(split)
            logger.info(f'Vocabulary has expanded to {numericalizer.num_tokens} tokens')

    for emb in embeddings:
        emb.grow_for_vocab(numericalizer.vocab, new_words)

    return splits


def run(args, numericalizer, val_sets, model, device):
    logger.info(f'Preparing iterators')
    if len(args.val_batch_size) == 1 and len(val_sets) > 1:
        args.val_batch_size *= len(val_sets)
    iters = []
    task_index = 0
    for task, bs, val_set in zip(args.tasks, args.val_batch_size, val_sets):
        task_iter = []
        task_languages = args.pred_languages[task_index]
        if task_languages is not None and args.separate_eval:
            task_languages = task_languages.split('+')
            assert len(task_languages) == len(val_set)
            for index, set in enumerate(val_set):
                loader = make_data_loader(set, numericalizer, bs, device,
                                          append_question_to_context_too=args.append_question_to_context_too,
                                          override_question=args.override_question)
                task_iter.append((task, task_languages[index], loader))
        # single language task or no separate eval
        else:
           loader = make_data_loader(val_set[0], numericalizer, bs, device,
                                     append_question_to_context_too=args.append_question_to_context_too,
                                     override_question=args.override_question)
           task_iter.append((task, task_languages, loader))

        iters.extend(task_iter)
        task_index += 1

    log_model_size(logger, model, args.model)
    model.to(device)

    decaScore = []
    task_scores = defaultdict(list)
    model.eval()

    eval_dir = os.path.join(args.eval_dir, args.evaluate)
    os.makedirs(eval_dir, exist_ok=True)

    with torch.no_grad():
        for task, language, it in iters:
            logger.info(task.name)
            # single language task
            if language is None:
                prediction_file_name = os.path.join(eval_dir, task.name + '.tsv')
                results_file_name = os.path.join(eval_dir, task.name + '.results.json')
            # multi language task
            else:
                prediction_file_name = os.path.join(eval_dir, task.name + '_{}.tsv'.format(language))
                results_file_name = os.path.join(eval_dir, task.name + '_{}.results.json'.format(language))
            if os.path.exists(prediction_file_name):
                if args.overwrite:
                    logger.warning(f'{prediction_file_name} already exists -- overwriting **')
                else:
                    raise OSError(f'{prediction_file_name} already exists')
            if os.path.exists(results_file_name):
                if args.overwrite:
                    logger.warning(f'{results_file_name} already exists -- overwriting **')
                else:
                    raise OSError(f'{results_file_name} already exists')

            predictions = []
            answers = []
            with open(prediction_file_name, 'w' + ('' if args.overwrite else 'x')) as prediction_file:
                for batch_idx, batch in tqdm(enumerate(it), desc="Batches"):
                    _, batch_prediction = model(batch, iteration=1)

                    batch_prediction = numericalizer.reverse(batch_prediction, detokenize=task.detokenize,
                                                             field_name='answer')
                    predictions += batch_prediction
                    batch_answer = numericalizer.reverse(batch.answer.value.data, detokenize=task.detokenize,
                                                         field_name='answer')
                    answers += batch_answer

                    for i, example_prediction in enumerate(batch_prediction):
                        prediction_file.write(batch.example_id[i] + '\t' + example_prediction + '\n')

            if len(answers) > 0:
                metrics, answers = compute_metrics(predictions, answers, task.metrics)
                with open(results_file_name, 'w' + ('' if args.overwrite else '+')) as results_file:
                    results_file.write(json.dumps(metrics) + '\n')

                if not args.silent:
                    for i, (p, a) in enumerate(zip(predictions, answers)):
                        logger.info(f'Prediction {i + 1}: {p}\nAnswer {i + 1}: {a}\n')
                    logger.info(metrics)
                    
                task_scores[task].append((len(answers), metrics[task.metrics[0]]))
    
    for task in task_scores.keys():
        decaScore.append(sum([lenght * score for lenght, score in task_scores[task]]) / sum([lenght for lenght, score in task_scores[task]]))

    logger.info(f'Evaluated Tasks:\n')
    for i, task in enumerate(args.tasks):
        logger.info(f'{task.name}: {decaScore[i]}')
    logger.info(f'-------------------')
    logger.info(f'DecaScore:  {sum(decaScore)}\n')
    logger.info(f'\nSummary: | {sum(decaScore)} | {" | ".join([str(x) for x in decaScore])} |\n')


def parse_argv(parser):
    parser.add_argument('--path', required=True)
    parser.add_argument('--evaluate', type=str, required=True, choices=['valid', 'test'],
                        help='Which dataset to do predictions for (test or dev)')
    parser.add_argument('--pred_set_name', type=str, help='Name of dataset to run prediction for; will be ignored if --evaluate is test')
    parser.add_argument('--tasks',
                        default=['almond', 'squad', 'iwslt.en.de', 'cnn_dailymail', 'multinli.in.out', 'sst', 'srl',
                                 'zre', 'woz.en', 'wikisql', 'schema'], dest='task_names', nargs='+')
    parser.add_argument('--devices', default=[0], nargs='+', type=int,
                        help='a list of devices that can be used (multi-gpu currently WIP)')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--embeddings', default='.embeddings/', type=str, help='where to save embeddings.')
    parser.add_argument('--checkpoint_name', default='best.pth',
                        help='Checkpoint file to use (relative to --path, defaults to best.pth)')
    parser.add_argument('--bleu', action='store_true', help='whether to use the bleu metric (always on for iwslt)')
    parser.add_argument('--rouge', action='store_true',
                        help='whether to use the bleu metric (always on for cnn, dailymail, and cnn_dailymail)')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite previously written predictions')
    parser.add_argument('--silent', action='store_true', help='whether to print predictions to stdout')

    parser.add_argument('--skip_cache', action='store_true',
                        help='whether use exisiting cached splits or generate new ones')
    parser.add_argument('--eval_dir', type=str, required=True, help='use this directory to store eval results')
    parser.add_argument('--cache', default='.cache', type=str, help='where to save cached files')

    parser.add_argument('--saved_models', default='./saved_models', type=str,
                        help='directory where cached models should be loaded from')
    parser.add_argument('--subsample', default=20000000, type=int,
                        help='subsample the eval/test datasets (experimental)')

    parser.add_argument('--pred_languages', type=str, nargs='+',
                        help='used to specify dataset languages used during prediction for multilingual tasks'
                        'multiple languages for each task should be concatenated with +')
    parser.add_argument('--separate_eval', action='store_true',
                        help='evaluate on each language eval set separately')


def adjust_multilingual_eval(args):
    if (have_multilingual(args.task_names) and args.pred_languages is None) or (
            args.pred_languages and len(args.task_names) != len(args.pred_languages)):
        raise ValueError('You have to define prediction languages when you have a multilingual task'
                         'Use None for single language tasks. Also provide languages in the same order you provided tasks.')

    if args.pred_languages is None:
        args.pred_languages = [None for _ in range(len(args.task_names))]

    # preserve backward compatibility for single language tasks
    for i, task_name in enumerate(args.task_names):
        if 'multilingual' in task_name and args.pred_languages[i] is None:
            raise ValueError('You have to define prediction languages for this multilingual task: {}'.format(task_name))
        elif 'multilingual' not in task_name and args.pred_languages[i] is not None:
            logger.warning('prediction languages should be empty for single language tasks')
            args.pred_languages[i] = None
            

def main(args):
    load_config_json(args)
    adjust_multilingual_eval(args)
    set_seed(args)
    args.tasks = get_tasks(args.task_names, args)

    logger.info(f'Arguments:\n{pformat(vars(args))}')
    logger.info(f'Loading from {args.best_checkpoint}')

    devices = init_devices(args)
    save_dict = torch.load(args.best_checkpoint, map_location=devices[0])

    numericalizer, context_embeddings, question_embeddings, decoder_embeddings = \
        load_embeddings(args.embeddings, args.context_embeddings, args.question_embeddings, args.decoder_embeddings,
                        args.max_generative_vocab, logger)
    numericalizer.load(args.path)
    for emb in set(context_embeddings + question_embeddings + decoder_embeddings):
        emb.init_for_vocab(numericalizer.vocab)

    logger.info(f'Initializing Model')
    Model = getattr(models, args.model)
    model = Model(numericalizer, args, context_embeddings, question_embeddings, decoder_embeddings)
    model_dict = save_dict['model_state_dict']
    model.load_state_dict(model_dict)
    splits = prepare_data(args, numericalizer, set(context_embeddings + question_embeddings + decoder_embeddings))

    run(args, numericalizer, splits, model, devices[0])
