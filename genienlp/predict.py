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

import torch

from . import models
from .data_utils.embeddings import load_embeddings
from .metrics import compute_metrics
from .tasks.registry import get_tasks
from .util import set_seed, preprocess_examples, load_config_json, make_data_loader, log_model_size, init_devices

logger = logging.getLogger(__name__)


def get_all_splits(args):
    splits = []
    for task in args.tasks:
        logger.info(f'Loading {task}')
        kwargs = {}
        if not 'train' in args.evaluate:
            kwargs['train'] = None
        if not 'valid' in args.evaluate:
            kwargs['validation'] = None
        if not 'test' in args.evaluate:
            kwargs['test'] = None

        kwargs['skip_cache'] = args.skip_cache
        kwargs['cached_path'] = os.path.join(args.cache, task.name)
        kwargs['subsample'] = args.subsample
        s = task.get_splits(root=args.data, lower=args.lower, **kwargs)[0]
        preprocess_examples(args, [task], [s], train=False)
        splits.append(s)
    return splits


def prepare_data(args, numericalizer, embeddings):
    splits = get_all_splits(args)
    logger.info(f'Vocabulary has {numericalizer.num_tokens} tokens from training')
    new_words = []
    for split in splits:
        new_words += numericalizer.grow_vocab(split)
        logger.info(f'Vocabulary has expanded to {numericalizer.num_tokens} tokens')

    for emb in embeddings:
        emb.grow_for_vocab(numericalizer.vocab, new_words)

    return splits


def run(args, numericalizer, val_sets, model, device):
    logger.info(f'Preparing iterators')
    if len(args.val_batch_size) == 1 and len(val_sets) > 1:
        args.val_batch_size *= len(val_sets)
    iters = [(name, make_data_loader(x, numericalizer, bs, device)) for name, x, bs in
             zip(args.tasks, val_sets, args.val_batch_size)]

    log_model_size(logger, model, args.model)
    model.to(device)

    decaScore = []
    model.eval()

    eval_dir = os.path.join(args.eval_dir, args.evaluate)
    os.makedirs(eval_dir, exist_ok=True)

    with torch.no_grad():
        for task, it in iters:
            logger.info(task.name)
            prediction_file_name = os.path.join(eval_dir, task.name + '.tsv')
            results_file_name = os.path.join(eval_dir, task.name + '.results.json')
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
                metrics, answers = compute_metrics(predictions, answers, task.metrics, args=args)
                with open(results_file_name, 'w' + ('' if args.overwrite else '+')) as results_file:
                    results_file.write(json.dumps(metrics) + '\n')

                if not args.silent:
                    for i, (p, a) in enumerate(zip(predictions, answers)):
                        logger.info(f'Prediction {i + 1}: {p}\nAnswer {i + 1}: {a}\n')
                    logger.info(metrics)
                decaScore.append(metrics[task.metrics[0]])

    logger.info(f'Evaluated Tasks:\n')
    for i, (task, _) in enumerate(iters):
        logger.info(f'{task.name}: {decaScore[i]}')
    logger.info(f'-------------------')
    logger.info(f'DecaScore:  {sum(decaScore)}\n')
    logger.info(f'\nSummary: | {sum(decaScore)} | {" | ".join([str(x) for x in decaScore])} |\n')


def parse_argv(parser):
    parser.add_argument('--path', required=True)
    parser.add_argument('--evaluate', type=str, required=True, choices=['valid', 'test'], help='Which dataset to evaluate (test or dev)')
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


def main(args):
    load_config_json(args)
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
