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
    parser.add_argument('--save', required=True, type=str, help='where to save results.')

    parser.add_argument(
        '--val_tasks', nargs='+', type=str, dest='val_task_names', help='tasks to collect evaluation metrics for'
    )

    parser.add_argument(
        '--val_batch_size',
        nargs='+',
        default=[4000],
        type=int,
        help='Number of tokens in each batch for validation, corresponding to tasks in --val_tasks',
    )

    parser.add_argument('--max_output_length', default=150, type=int, help='maximum output length for generation')
    parser.add_argument(
        '--min_output_length',
        default=3,
        type=int,
        help='maximum output length for generation; '
        'default is 3 for most multilingual models: BOS, language code, and one token. otherwise it is 2',
    )
    parser.add_argument('--max_generative_vocab', default=50000, type=int, help='max vocabulary for the generative softmax')
    parser.add_argument(
        '--filter_long_inputs',
        action='store_true',
        help='Filter out examples that are longer than required model input_max_length',
    )
    parser.add_argument('--preserve_case', action='store_false', dest='lower', help='whether to preserve casing for all text')
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

    parser.add_argument(
        '--pretrained_model',
        default=None,
        help='which pretrained model to use on the encoder side; choose a name from Huggingface models',
    )

    parser.add_argument(
        '--num_workers', type=int, default=0, help='Number of processes to use for data loading (0 means no multiprocessing)'
    )

    parser.add_argument('--override_context', type=str, default=None, help='Override the context for all tasks')
    parser.add_argument('--override_question', type=str, default=None, help='Override the question for all tasks')
    parser.add_argument(
        "--almond_has_single_program",
        action='store_false',
        dest='almond_has_multiple_programs',
        help='Indicate if almond dataset has multiple programs for each sentence',
    )
    parser.add_argument(
        '--almond_lang_as_question',
        action='store_true',
        help='if true will use "Translate from ${language} to ThingTalk" for question',
    )
    parser.add_argument(
        '--almond_detokenize_sentence',
        action='store_true',
        help='undo word tokenization of almond sentence fields (useful if the tokenizer is sentencepiece)',
    )
    parser.add_argument('--preprocess_special_tokens', action='store_true', help='convert special ThingTalk tokens to words')

    parser.add_argument(
        '--load',
        default=None,
        type=str,
        help='path to checkpoint to load model from inside --args.save, usually set to best.pth',
    )
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')

    parser.add_argument(
        '--no_commit',
        action='store_false',
        dest='commit',
        help='do not track the git commit associated with this training run',
    )

    parser.add_argument(
        '--no_fast_tokenizer', action='store_true', help='Ignore all conditions and use slow version of huggingface tokenizer'
    )
    parser.add_argument(
        '--force_fast_tokenizer',
        action='store_true',
        help='Ignore all conditions and use fast version of huggingface tokenizer',
    )

    parser.add_argument('--use_curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument(
        '--aux_dataset', default='', type=str, help='path to auxiliary dataset (ignored if curriculum is not used)'
    )
    parser.add_argument(
        '--curriculum_max_frac', default=1.0, type=float, help='max fraction of harder dataset to keep for curriculum'
    )
    parser.add_argument('--curriculum_rate', default=0.1, type=float, help='growth rate for curriculum')
    parser.add_argument(
        '--curriculum_strategy', default='linear', type=str, choices=['linear', 'exp'], help='growth strategy for curriculum'
    )
    parser.add_argument(
        '--database_dir',
        type=str,
        default='database/',
        help='Database folder containing all relevant files (e.g. alias2qids, pretrained models for bootleg)',
    )

    parser.add_argument(
        "--almond_type_mapping_path",
        default=None,
        type=str,
        help='If provided, will override the usual almond type mapping in data_utils/database_file/'
        'Path should be relative to --root',
    )

    # translation args
    parser.add_argument(
        '--att_pooling',
        type=str,
        default='max',
        help='pooling strategy to calculate cross-attention values across multiple heads',
    )
    parser.add_argument('--plot_heatmaps', action='store_true', help='whether to plot cross-attention heatmaps')
    parser.add_argument(
        '--do_alignment',
        action='store_true',
        help='whether to preserve token spans between quotation marks using alignment during translation',
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
        default='"',
        type=str,
        help='The symbol we use to wrap spans of words in the input that need to be preserved in the output.',
    )

    # token classification task args
    parser.add_argument('--num_labels', type=int, help='num_labels for classification tasks')
    parser.add_argument('--crossner_domains', nargs='+', type=str, help='domains to use for CrossNER task')

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
    parser.add_argument(
        '--e2e_dialogue_valid_subweights',
        nargs='+',
        type=float,
        default=[1.0, 1.0, 1.0],
        help='Specify weights to use for each of subtasks in e2e_dialogue_valid_subtasks.',
    )


def check_and_update_generation_args(args):
    """
    checks all generation commandline arguments. Since these arguments are all lists and shorthand can be used, we expand them to match the expected length
    for instance, [1.0] becomes [1.0 1.0] if all other generation arguments are of length 2
    """
    hyperparameters = [
        'num_outputs',
        'temperature',
        'top_k',
        'top_p',
        'repetition_penalty',
        'num_beams',
        'num_beam_groups',
        'diversity_penalty',
        'no_repeat_ngram_size',
    ]
    max_hyperparameter_len = max([len(getattr(args, h)) for h in hyperparameters])
    valid_len = [1, max_hyperparameter_len]
    for h in hyperparameters:
        if len(getattr(args, h)) not in valid_len:
            logger.error('Hyperparameters should either have the same number of values as others or have exactly one value.')
        # If only one value is provided, use the same value for all samples
        setattr(args, h, getattr(args, h) * (max_hyperparameter_len // len(getattr(args, h))))

    logger.info('Will output %d sequences for each input.', sum(args.num_outputs))
    return args


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
    if args.e2e_dialogue_evaluation and args.val_batch_size[0] != 1:
        logger.warning('When evaluating bitod end2end val_batch_size should be 1 so we load data turn by turn')
        args.val_batch_size = [1]

    if len(args.e2e_dialogue_valid_subtasks) != len(args.e2e_dialogue_valid_submetrics):
        raise ValueError(
            'Length of e2e_dialogue_valid_subtasks and e2e_dialogue_valid_submetrics arguments should be equal (i.e. one metric per subtask)'
        )

    if len(args.e2e_dialogue_valid_subtasks) != len(args.e2e_dialogue_valid_subweights):
        raise ValueError(
            'Length of e2e_dialogue_valid_subtasks and e2e_dialogue_valid_subweights arguments should be equal (i.e. one weight per subtask)'
        )

    if len(args.val_batch_size) < len(args.val_task_names):
        args.val_batch_size = len(args.val_task_names) * args.val_batch_size

    if args.no_fast_tokenizer and args.force_fast_tokenizer:
        raise ValueError('Both no_fast_tokenizer and force_fast_tokenizer flags are on')

    if args.override_valid_metrics:
        assert len(args.override_valid_metrics) == len(args.train_tasks) == len(args.val_tasks)
        for train_task, val_task, metrics in zip(args.train_tasks, args.val_tasks, args.override_valid_metrics):
            train_task.metrics = metrics
            val_task.metrics = metrics

    args.log_dir = args.save
    if args.tensorboard_dir is None:
        args.tensorboard_dir = args.log_dir

    save_args(args)

    args = check_and_update_generation_args(args)
    return args
