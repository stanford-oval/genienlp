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

from .metrics import calculate_metrics
from .models.base import ValidationOutput
from .tasks.registry import get_task

logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument(
        '--pred_file',
        required=True,
        type=str,
        help='Name of dataset to run prediction for; will be ignored if --evaluate is test',
    )
    parser.add_argument('--task', dest='task_name', required=True, help='task names for prediction')
    parser.add_argument('--llm', type=str, choices=["gpt-4", "gpt-35-turbo"], required=True, help='The LLM to use for inference')
    parser.add_argument('--llm_url', type=str, required=True, help='The LLM inference server URL')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the prediction file')

    parser.add_argument('--eval_dir', type=str, required=False, help='use this directory to store eval results')

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


def compute_metrics_on_file(pred_file, results_file_name, task, args):
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
    metrics = calculate_metrics(
        args,
        validation_output,
        metrics_to_compute,
    )

    with open(results_file_name, 'w') as results_file:
        results_file.write(json.dumps(metrics) + '\n')

    logger.info(metrics)


def main(args):
    args.task = get_task(args.task_name, args)

    logger.info(f'Arguments:\n{pformat(vars(args))}')

    if not args.eval_dir:
        eval_dir = os.path.dirname(args.pred_file)
    else:
        eval_dir = args.eval_dir
    os.makedirs(eval_dir, exist_ok=True)

    results_file_name = os.path.join(eval_dir, args.task.name + '.results.json')
    compute_metrics_on_file(args.pred_file, results_file_name, args.task, args)
