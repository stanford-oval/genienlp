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

import torch

from genienlp.models import TransformerSeq2Seq

from .arguments import check_and_update_generation_args
from .metrics import calculate_and_reduce_metrics
from .tasks.registry import get_tasks
from .util import (
    load_config_file_to_args,
    make_data_loader,
    set_seed,
)

logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument('--path', '--model_name_or_path', type=str, required=True, help='Folder to load the model from')
    parser.add_argument(
        '--evaluate',
        type=str,
        required=True,
        choices=['train', 'valid', 'test'],
        help='Which dataset to do predictions for (train, dev or test)',
    )
    parser.add_argument(
        '--pred_set_name',
        default='eval',
        type=str,
        help='Name of dataset to run prediction for; will be ignored if --evaluate is test',
    )
    parser.add_argument('--tasks', dest='task_names', nargs='+', help='task names for prediction')
    parser.add_argument('--extra_metrics', nargs='+', default=[], help='include these additional metrics in reported results')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument(
        '--checkpoint_name', default='best.pth', help='Checkpoint file to use (relative to --path, defaults to best.pth)'
    )
    parser.add_argument('--silent', action='store_true', help='whether to print predictions to stdout')
    parser.add_argument('--language', type=str, required=True, help='The language of the output, used for postprocessing.')

    parser.add_argument('--eval_dir', type=str, required=True, help='use this directory to store eval results')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the eval/test datasets')

    parser.add_argument(
        '--main_metric_only', action='store_true', help='If True, we only calculate the deca score metric for each task.'
    )
    # If not None, these values will override the values saved in the trained model's config file
    parser.add_argument(
        '--val_batch_size',
        nargs='+',
        default=None,
        type=int,
        help='Batch size for validation corresponding to tasks in val tasks',
    )
    parser.add_argument(
        "--reduce_metrics",
        type=str,
        default='max',
        choices=['max', 'top_k'],
        help='How to calculate the metric when there are multiple outputs per input.'
        '`max` chooses the best set of generation hyperparameters and reports the metric for that.'
        '`top_k` chooses the best generation output per input, and uses that to output the metric. For example, combining this with the exact match metric gives what is commonly known as the top-k accuracy. Note that the output is meaningless if used with corpus-level metrics.',
    )

    # These are generation hyperparameters. Each one can be a list of values in which case, we generate `num_outputs` outputs for each set of hyperparameters.
    parser.add_argument("--num_outputs", type=int, nargs='+', default=[1], help='number of sequences to output per input')
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
    parser.add_argument("--num_beams", type=int, nargs='+', default=[1], help='1 disables beam seach')
    parser.add_argument("--num_beam_groups", type=int, nargs='+', default=[1], help='1 disables diverse beam seach')
    parser.add_argument("--diversity_penalty", type=float, nargs='+', default=[0.0], help='0 disables diverse beam seach')
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        nargs='+',
        default=[0],
        help='ngrams of this size cannot be repeated in the output. 0 disables it.',
    )
    parser.add_argument('--max_output_length', type=int, help='maximum output length for generation')
    parser.add_argument(
        '--min_output_length',
        type=int,
        help='maximum output length for generation; '
        'default is 3 for most multilingual models: BOS, language code, and one token. otherwise it is 2',
    )

    parser.add_argument(
        '--database_dir', type=str, help='Path to folder containing all files (e.g. alias2qids, pretrained models for bootleg)'
    )

    parser.add_argument(
        '--one_output_per_line',
        action='store_true',
        help='If true, each of the `num_outputs` output will be written to a separate line, while other columns are duplicated to fill these extra lines.',
    )

    parser.add_argument(
        '--filter_long_inputs',
        action='store_true',
        help='Filter out examples that are longer than required model input_max_length',
    )

    parser.add_argument('--plot_heatmaps', action='store_true', help='whether to plot cross-attention heatmaps')
    parser.add_argument(
        '--do_alignment',
        action='store_true',
        help='whether to replace tokens between quotation marks after translation with source values',
    )
    parser.add_argument(
        '--align_preserve_input_quotation',
        action='store_true',
        help='preserve quotation marks in the input. Useful if using alignment for semantic parsing or NLG',
    )
    parser.add_argument(
        '--align_remove_output_quotation',
        action='store_true',
        help='do not preserve quotation marks in the output. Useful if using alignment for semantic parsing or NLG',
    )
    parser.add_argument(
        '--align_span_symbol',
        type=str,
        help='The symbol we use to wrap spans of words in the input that need to be preserved in the output.',
    )

    parser.add_argument(
        '--e2e_dialogue_evaluation',
        action='store_true',
        help='Evaluate model on a dialogue dataset end-to-end; i.e. model predictions are used as input instead of gold',
    )
    parser.add_argument(
        '--e2e_dialogue_valid_subtasks',
        nargs='+',
        type=str,
        help='Evaluate only on these subtasks when calculating e2e_dialogue_score; rg is not included by default',
    )
    parser.add_argument(
        '--e2e_dialogue_valid_submetrics',
        nargs='+',
        type=str,
        help='Specify metrics to use for each of subtasks in e2e_dialogue_valid_subtasks.',
    )
    parser.add_argument(
        '--e2e_dialogue_valid_subweights',
        nargs='+',
        type=float,
        help='Specify weights to use for each of subtasks in e2e_dialogue_valid_subtasks.',
    )

    parser.add_argument(
        '--align_helper_file',
        type=str,
        help='dictionary path',
    )


def set_default_values(args):
    """
    sets default values that depend on other input arguments
    """
    if args.e2e_dialogue_evaluation and args.val_batch_size[0] != 1:
        logger.warning('When evaluating dialogues end-to-end, val_batch_size should be 1 so we load the data turn by turn')
        args.val_batch_size = [1]


def check_args(args):
    if args.main_metric_only and args.extra_metrics:
        raise ValueError('Please remove --main_metric_only from your arguments so the requested extra metrics can be shown.')


def prepare_data(args):

    datasets = []
    paths = []
    for i, task in enumerate(args.tasks):
        logger.info(f'Loading {task}')
        kwargs = {'train': None, 'validation': None, 'test': None}
        if args.evaluate == 'train':
            del kwargs['train']  # deleting keys means use the default file name
        elif args.evaluate == 'valid':
            kwargs['validation'] = args.pred_set_name
        elif args.evaluate == 'test':
            del kwargs['test']
        else:
            raise ValueError('Split used for prediction should be either train, valid or test')

        kwargs.update(
            {
                'subsample': args.subsample,
                'num_workers': args.num_workers,
            }
        )

        split, path = task.get_splits(root=args.data, lower=args.lower, **kwargs)
        assert (split.eval or split.test or split.train) and not split.aux
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


def prepare_data_iterators(args, val_sets, numericalizer):
    logger.info('Preparing data iterators')
    if len(args.val_batch_size) == 1 and len(val_sets) > 1:
        args.val_batch_size *= len(val_sets)
    iters = []
    for task, bs, val_set in zip(args.tasks, args.val_batch_size, val_sets):
        task_iter = []
        loader, original_order = make_data_loader(val_set, numericalizer, bs, train=False, return_original_order=True)
        task_iter.append((task, loader, original_order))

        iters.extend(task_iter)

    return iters


def create_output_lines(args, index, validation_output):
    predictions = validation_output.predictions

    if args.one_output_per_line:
        lines = [
            '\t'.join(
                [
                    validation_output.example_ids[index],
                    prediction,
                    validation_output.answers[index],
                    validation_output.contexts[index],
                ]
            )
            for prediction in predictions[index]
        ]  # one line per generation output
    else:
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


def get_metrics_to_compute(args, task):
    metrics_to_compute = task.metrics
    metrics_to_compute += args.extra_metrics
    metrics_to_compute = [metric for metric in task.metrics if metric not in ['loss']]
    if args.main_metric_only:
        metrics_to_compute = [metrics_to_compute[0]]
    return metrics_to_compute


def run(args):
    model = TransformerSeq2Seq.load(
        args.path,
        args=args,
    )
    val_sets = prepare_data(args)
    iters = prepare_data_iterators(args, val_sets, model.numericalizer)

    task_scores = defaultdict(list)

    eval_dir = os.path.join(args.eval_dir, args.evaluate)
    os.makedirs(eval_dir, exist_ok=True)

    for (task, it, original_order) in iters:
        logger.info(task.name)
        prediction_file_name = os.path.join(eval_dir, task.name + '.tsv')
        results_file_name = os.path.join(eval_dir, task.name + '.results.json')

        validation_output = model.validate(
            it,
            task,
            eval_dir=eval_dir,
            original_order=original_order,
        )

        # write into file
        with open(prediction_file_name, 'w') as prediction_file:
            for i in range(len(validation_output.example_ids)):
                lines = create_output_lines(args, i, validation_output)
                prediction_file.write('\n'.join(lines) + '\n')

        if len(validation_output.answers) > 0:
            metrics_to_compute = get_metrics_to_compute(args, task)
            metrics = calculate_and_reduce_metrics(args, validation_output, metrics_to_compute)

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


def update_metrics(args):
    assert len(args.override_valid_metrics) == len(args.tasks)
    new_metrics = []
    for task, metrics in zip(args.tasks, args.override_valid_metrics):
        for m in metrics:
            # remove loss from validation metrics
            if m == 'loss':
                continue
            # backward compatibility for models validated on sacrebleu (now casedbleu)
            if m == 'sacrebleu':
                m = 'casedblue'
            new_metrics.append(m)

        task.metrics = new_metrics


def main(args):
    load_config_file_to_args(args)
    check_and_update_generation_args(args)
    check_args(args)
    set_default_values(args)

    set_seed(args)
    args.tasks = list(get_tasks(args.task_names, args).values())
    if args.override_valid_metrics:
        update_metrics(args)

    logger.info(f'Arguments:\n{pformat(vars(args))}')
    logger.info(f'Loading from {args.best_checkpoint}')

    run(args)
