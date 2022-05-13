#
# Copyright (c) 2022, The Board of Trustees of the Leland Stanford Junior University
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

from .metrics import calculate_and_reduce_metrics
from .models.base import ValidationOutput
from .tasks.registry import get_tasks
from .util import set_seed

logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument(
        '--pred_file',
        required=True,
        type=str,
        help='Name of dataset to run prediction for; will be ignored if --evaluate is test',
    )
    parser.add_argument('--tasks', dest='task_names', nargs='+', required=True, help='task names for prediction')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite previously written predictions')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the prediction file')

    parser.add_argument('--eval_dir', type=str, required=False, help='use this directory to store eval results')

    parser.add_argument(
        '--pred_languages',
        type=str,
        nargs='+',
        dest='pred_src_languages',
        default=['en'],
        help='Specify dataset source languages used during prediction for multilingual tasks',
    )
    parser.add_argument(
        '--pred_tgt_languages',
        type=str,
        nargs='+',
        default=['en'],
        help='Specify dataset target languages used during prediction for multilingual tasks',
    )

    parser.add_argument(
        '--main_metric_only', action='store_true', help='If True, we only calculate the deca score metric for each task.'
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
    parser.add_argument('--extra_metrics', nargs='+', default=[], help='include these additional metrics in reported results')

    parser.add_argument(
        '--e2e_dialogue_valid_subtasks',
        nargs='+',
        type=str,
        default=['dst', 'api', 'da', 'rg'],
        help='Evaluate only on these subtasks when calculating e2e_dialogue_score; rg is not included by default',
    )
    parser.add_argument(
        '--e2e_dialogue_valid_submetrics',
        nargs='+',
        type=str,
        default=['em', 'em', 'em', 'casedbleu'],
        help='Specify metrics to use for each of subtasks in e2e_dialogue_valid_subtasks.',
    )
    parser.add_argument(
        '--e2e_dialogue_valid_subweights',
        nargs='+',
        type=float,
        default=[1.0, 1.0, 1.0, 1.0],
        help='Specify weights to use for each of subtasks in e2e_dialogue_valid_subtasks.',
    )


def compute_metrics_on_file(pred_file, results_file_name, task, args, tgt_lang):
    ids, contexts, preds, targets = [], [], [], []
    count = 0
    with open(pred_file) as fin:
        for line in fin:
            id_, *pred, target, context = line.strip('\n').split('\t')
            ids.append(id_)
            contexts.append(context)
            preds.append(pred)
            targets.append(target)
            count += 1
            if count >= args.subsample:
                break

    validation_output = ValidationOutput(example_ids=ids, contexts=contexts, predictions=preds, answers=targets)

    metrics_to_compute = task.metrics
    metrics_to_compute += args.extra_metrics
    metrics_to_compute = [metric for metric in task.metrics if metric not in ['loss']]
    if args.main_metric_only:
        metrics_to_compute = [metrics_to_compute[0]]
    metrics = calculate_and_reduce_metrics(
        args,
        validation_output,
        metrics_to_compute,
        tgt_lang,
    )

    with open(results_file_name, 'w' + ('' if args.overwrite else '+')) as results_file:
        results_file.write(json.dumps(metrics) + '\n')

    logger.info(metrics)


def main(args):
    # these are needed when initializing the tasks
    args.override_context = None
    args.override_question = None
    args.almond_has_multiple_programs = None
    args.almond_detokenize_sentence = None
    args.do_alignment = None

    if args.main_metric_only and args.extra_metrics:
        raise ValueError('Please remove --main_metric_only from your arguments so the requested extra metrics can be shown.')

    set_seed(args)
    args.tasks = list(get_tasks(args.task_names, args).values())

    logger.info(f'Arguments:\n{pformat(vars(args))}')

    if not args.eval_dir:
        eval_dir = os.path.dirname(args.pred_file)
    else:
        eval_dir = args.eval_dir
    os.makedirs(eval_dir, exist_ok=True)
    tgt_lang = args.pred_tgt_languages[0]

    for task in args.tasks:
        results_file_name = os.path.join(eval_dir, task.name + '.results.json')
        compute_metrics_on_file(args.pred_file, results_file_name, task, args, tgt_lang)
