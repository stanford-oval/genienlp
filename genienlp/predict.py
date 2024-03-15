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
from collections import defaultdict
from pprint import pformat


from genienlp.models import TransformerSeq2Seq

from .metrics import calculate_metrics
from .tasks.registry import get_tasks
from .util import (
    make_data_loader,
)

logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument(
        '--evaluate',
        type=str,
        required=True,
        choices=['train', 'valid', 'test'],
        help='Which dataset to do predictions for (train, dev or test)',
    )
    parser.add_argument('--tasks', dest='task_names', nargs='+', help='task names for prediction')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--silent', action='store_true', help='whether to print predictions to stdout')
    parser.add_argument('--language', type=str, required=True, help='The language of the output, used for postprocessing.')

    parser.add_argument('--eval_dir', type=str, required=True, help='use this directory to store eval results')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the eval/test datasets')

    # If not None, these values will override the values saved in the trained model's config file
    parser.add_argument(
        '--val_batch_size',
        default=40,
        type=int,
        help='Batch size for validation corresponding to tasks in val tasks',
    )

    parser.add_argument("--temperature", type=float, nargs='+', default=[0.0], help="temperature of 0 implies greedy sampling")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        nargs='+',
        default=[1.0],
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[1.0], help='1.0 disables top-p filtering')
    parser.add_argument('--output_length', type=int, help='maximum output length for generation')

    parser.add_argument(
        '--e2e_dialogue_evaluation',
        action='store_true',
        help='Evaluate model on a dialogue dataset end-to-end; i.e. model predictions are used as input instead of gold',
    )
    parser.add_argument(
        '--e2e_dialogue_valid_subtasks',
        nargs='+',
        type=str,
        default=['dst', 'api', 'da'],
        help='Evaluate only on these subtasks when calculating e2e_dialogue_score; rg is not included by default',
    )
    parser.add_argument(
        '--e2e_dialogue_valid_submetrics',
        nargs='+',
        type=str,
        default=['dst_em', 'em', 'da_em'],
        help='Specify metrics to use for each of subtasks in e2e_dialogue_valid_subtasks.',
    )


def set_default_values(args):
    """
    sets default values that depend on other input arguments
    """
    if args.e2e_dialogue_evaluation and args.val_batch_size != 1:
        logger.warning('When evaluating dialogues end-to-end, val_batch_size should be 1 so we load the data turn by turn')
        args.val_batch_size = 1


def prepare_data(args):

    datasets = []
    paths = []
    for i, task in enumerate(args.tasks):
        logger.info(f'Loading {task}')
        kwargs = {'train': None, 'validation': None, 'test': None}
        if args.evaluate == 'train':
            del kwargs['train']  # deleting keys means use the default file name
        elif args.evaluate == 'valid':
            kwargs['validation'] = 'valid'
        elif args.evaluate == 'test':
            del kwargs['test']
        else:
            raise ValueError('Split used for prediction should be either train, valid or test')

        kwargs.update(
            {
                'subsample': args.subsample,
            }
        )

        split, path = task.get_splits(root=args.data, **kwargs)
        if split.train:
            data = split.train
            path = path.train
        elif split.eval:
            data = split.eval
            path = path.eval
        else:
            data = split.test
            path = path.test

        logger.info(f'{task.name} has {len(data.examples)} prediction examples')
        datasets.append(data)
        paths.append(path)

    return datasets


def prepare_data_iterators(args, val_sets):
    logger.info('Preparing data iterators')
    iters = []
    for task, bs, val_set in zip(args.tasks, [args.val_batch_size] * len(val_sets), val_sets):
        task_iter = []
        loader = make_data_loader(val_set, bs)
        task_iter.append((task, loader))

        iters.extend(task_iter)

    return iters


def create_output_lines(index, validation_output):
    predictions = validation_output.predictions
    lines = [
        '\t'.join(
            [
                validation_output.example_ids[index],
                *predictions[index],
                validation_output.answers[index],
                validation_output.contexts[index],
            ]
        )
    ]  # one line with all generation outputs separated by '\t'
    return lines


def run(args):
    model = TransformerSeq2Seq.load(
        args=args,
    )
    val_sets = prepare_data(args)
    iters = prepare_data_iterators(args, val_sets)

    task_scores = defaultdict(list)

    eval_dir = os.path.join(args.eval_dir, args.evaluate)
    os.makedirs(eval_dir, exist_ok=True)

    for task, it in iters:
        logger.info(task.name)
        prediction_file_name = os.path.join(eval_dir, task.name + '.tsv')
        results_file_name = os.path.join(eval_dir, task.name + '.results.json')

        validation_output = model.validate(
            it,
            task,
            eval_dir=eval_dir,
        )

        # write into file
        with open(prediction_file_name, 'w') as prediction_file:
            for i in range(len(validation_output.example_ids)):
                lines = create_output_lines(i, validation_output)
                prediction_file.write('\n'.join(lines) + '\n')

        if len(validation_output.answers) > 0:
            metrics = calculate_metrics(args, validation_output, task)

            with open(results_file_name, 'w') as results_file:
                results_file.write(json.dumps(metrics) + '\n')

            if not args.silent:
                for i, (c, p, a) in enumerate(
                    zip(validation_output.contexts, validation_output.predictions, validation_output.answers)
                ):
                    log_string = '\n'.join(
                        [f'Context {i + 1}: {c}', f'Prediction {i + 1} ({len(p)} outputs): {p}', f'Answer {i + 1}: {a}']
                    )
                    logger.info(log_string)

            logger.info(metrics)

            task_scores[task].append((len(validation_output.answers), metrics[task.metrics[0]]))

    decaScore = []
    for task in task_scores.keys():
        decaScore.append(
            sum([length * score for length, score in task_scores[task]]) / sum([length for length, score in task_scores[task]])
        )

    logger.info('Evaluated Tasks:\n')
    for i, task in enumerate(args.tasks):
        logger.info(f'{task.name}: {decaScore[i]}')
    logger.info('-------------------')
    logger.info(f'DecaScore:  {sum(decaScore)}\n')
    logger.info(f'\nSummary: | {sum(decaScore)} | {" | ".join([str(x) for x in decaScore])} |\n')


def main(args):
    set_default_values(args)

    args.tasks = list(get_tasks(args.task_names, args).values())

    logger.info(f'Arguments:\n{pformat(vars(args))}')

    run(args)
