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

import datetime
import json
import logging
import os
import subprocess

from .tasks.registry import get_tasks

logger = logging.getLogger(__name__)


def get_commit():
    directory = os.path.dirname(__file__)
    return (
        subprocess.Popen("cd {} && git log | head -n 1".format(directory), shell=True, stdout=subprocess.PIPE)
        .stdout.read()
        .split()[1]
        .decode()
    )


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=True)
    variables = vars(args).copy()
    # remove the task objects before saving the configuration to the JSON file,
    # because tasks are not JSON serializable.
    del variables['train_tasks']
    del variables['val_tasks']
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(variables, f, indent=2)


def parse_argv(parser):
    parser.add_argument('--root', default='.', type=str, help='root directory for data, results, code, etc.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')

    parser.add_argument(
        '--val_tasks', nargs='+', type=str, dest='val_task_names', help='tasks to collect evaluation metrics for'
    )

    parser.add_argument(
        '--val_batch_size',
        default=40,
        type=int,
        help='Number of tokens in each batch for validation, corresponding to tasks in --val_tasks',
    )

    parser.add_argument('--max_output_length', default=150, type=int, help='maximum output length for generation')

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


def post_parse_general(args):
    if args.val_task_names is None:
        args.val_task_names = []

    args.timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime('%D-%H:%M:%S %Z')

    # postprocess arguments
    if args.commit:
        args.commit = get_commit()
    else:
        args.commit = ''


    args.log_dir = args.save
    args.dist_sync_file = os.path.join(args.log_dir, 'distributed_sync_file')

    for x in ['data', 'save', 'log_dir', 'dist_sync_file']:
        setattr(args, x, os.path.join(args.root, getattr(args, x)))

    # tasks with the same name share the same task object
    val_task_dict = get_tasks(args.val_task_names, args)
    args.val_tasks = list(val_task_dict.values())

    save_args(args)

    return args


def post_parse_train_specific(args):
    if args.e2e_dialogue_evaluation and args.val_batch_size != 1:
        logger.warning('When evaluating bitod end2end val_batch_size should be 1 so we load data turn by turn')
        args.val_batch_size = 1

    if len(args.e2e_dialogue_valid_subtasks) != len(args.e2e_dialogue_valid_submetrics):
        raise ValueError(
            'Length of e2e_dialogue_valid_subtasks and e2e_dialogue_valid_submetrics arguments should be equal (i.e. one metric per subtask)'
        )

    args.log_dir = args.save
    if args.tensorboard_dir is None:
        args.tensorboard_dir = args.log_dir

    save_args(args)

    return args
