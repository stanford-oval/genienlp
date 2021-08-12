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

from .model_utils.transformers_utils import MODEL_PARALLEL_SUPPORTED_MODELS
from .tasks.registry import get_tasks
from .util import have_multilingual

logger = logging.getLogger(__name__)


def get_commit():
    directory = os.path.dirname(__file__)
    return (
        subprocess.Popen("cd {} && git log | head -n 1".format(directory), shell=True, stdout=subprocess.PIPE)
        .stdout.read()
        .split()[1]
        .decode()
    )


def save_args(args, force_overwrite=False):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok or force_overwrite)
    variables = vars(args).copy()
    # remove the task objects before saving the configuration to the JSON file,
    # because tasks are not JSON serializable.
    del variables['train_tasks']
    del variables['val_tasks']
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(variables, f, indent=2)


def parse_argv(parser):
    parser.add_argument('--root', default='.', type=str, help='root directory for data, results, embeddings, code, etc.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--save', required=True, type=str, help='where to save results.')
    parser.add_argument('--embeddings', default='.embeddings/', type=str, help='where to save embeddings.')
    parser.add_argument('--cache', default='.cache/', type=str, help='where to save cached files')

    parser.add_argument(
        '--train_languages',
        type=str,
        default='en',
        dest='train_src_languages',
        help='Specify dataset source languages used during training for multilingual tasks'
        'multiple languages for each task should be concatenated with +',
    )
    parser.add_argument(
        '--eval_languages',
        type=str,
        default='en',
        dest='eval_src_languages',
        help='Specify dataset source languages used during validation for multilingual tasks'
        'multiple languages for each task should be concatenated with +',
    )

    parser.add_argument(
        '--train_tgt_languages',
        type=str,
        default='en',
        help='Specify dataset target languages used during training for multilingual tasks'
        'multiple languages for each task should be concatenated with +',
    )
    parser.add_argument(
        '--eval_tgt_languages',
        type=str,
        default='en',
        help='Specify dataset target languages used during validation for multilingual tasks'
        'multiple languages for each task should be concatenated with +',
    )

    parser.add_argument('--max_qids_per_entity', type=int, default=1, help='maximum number of qids to keep for each entity')
    parser.add_argument(
        '--max_types_per_qid',
        type=int,
        default=2,
        help='maximum number of types to keep for each qid associated with an entity',
    )

    parser.add_argument(
        '--train_tasks', nargs='+', type=str, dest='train_task_names', help='tasks to use for training', required=True
    )
    parser.add_argument('--train_iterations', nargs='+', type=int, help='number of iterations to focus on each task')
    # TODO rename to train_batch_size; keeping it for now for backward compatibility
    parser.add_argument(
        '--train_batch_tokens',
        nargs='+',
        default=[2000],
        type=int,
        help='Number of tokens to use for dynamic batching, corresponding to tasks in train tasks.'
        'If sentence_batching is used, this will be interpreted as number of examples.',
    )
    parser.add_argument('--jump_start', default=0, type=int, help='number of iterations to give jump started tasks')
    parser.add_argument('--n_jump_start', default=0, type=int, help='how many tasks to jump start (presented in order)')
    parser.add_argument(
        '--num_print', default=10, type=int, help='how many validation examples with greedy output to print to std out'
    )

    parser.add_argument(
        '--override_valid_metrics',
        nargs='+',
        action='append',
        type=str,
        help='if specified, will override metrics provided by the task (format is a list of lists)',
    )

    parser.add_argument(
        '--print_train_examples_too',
        action='store_true',
        help='Whether to print some train examples along with eval examples during validation',
    )

    parser.add_argument('--no_tensorboard', action='store_false', dest='tensorboard', help='Turn off tensorboard logging')
    parser.add_argument(
        '--tensorboard_dir', default=None, help='Directory where to save Tensorboard logs (defaults to --save)'
    )
    parser.add_argument('--max_to_keep', default=1, type=int, help='number of checkpoints to keep')
    parser.add_argument('--log_every', default=100, type=int, help='how often to log results in # of iterations')
    parser.add_argument('--save_every', default=1000, type=int, help='how often to save a checkpoint in # of iterations')

    parser.add_argument(
        '--val_tasks', nargs='+', type=str, dest='val_task_names', help='tasks to collect evaluation metrics for'
    )
    parser.add_argument('--val_every', default=1000, type=int, help='how often to run validation in # of iterations')
    parser.add_argument(
        '--val_batch_size',
        nargs='+',
        default=[4000],
        type=int,
        help='Number of tokens in each batch for validation, corresponding to tasks in --val_tasks',
    )

    parser.add_argument(
        '--sentence_batching', action='store_true', help='Batch same sentences together (used for multilingual tasks)'
    )
    parser.add_argument(
        '--use_encoder_loss',
        action='store_true',
        help='Force encoded values for sentences in different languages to be the same',
    )
    parser.add_argument(
        '--encoder_loss_type',
        type=str,
        default='mean',
        choices=['mean', 'sum'],
        help='Function to calculate encoder_loss from the context hidden states',
    )
    parser.add_argument(
        '--encoder_loss_weight',
        type=float,
        default=0.1,
        help='multiplicative constant choosing the weight of encoder_loss in total loss',
    )
    parser.add_argument('--eval_set_name', type=str, help='Evaluation dataset name to use during training')

    parser.add_argument('--max_output_length', default=150, type=int, help='maximum output length for generation')
    parser.add_argument('--max_generative_vocab', default=50000, type=int, help='max vocabulary for the generative softmax')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the datasets')
    parser.add_argument('--preserve_case', action='store_false', dest='lower', help='whether to preserve casing for all text')
    parser.add_argument(
        "--reduce_metrics",
        type=str,
        default='max',
        choices=['max'],
        help='How to calculate the metric when there are multiple outputs per input.',
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
        '--model',
        type=str,
        choices=[
            'TransformerLSTM',
            'TransformerSeq2Seq',
            'TransformerForTokenClassification',
            'TransformerForSequenceClassification',
        ],
        default='TransformerSeq2Seq',
        help='which model to import',
    )
    parser.add_argument(
        '--pretrained_model',
        default=None,
        help='which pretrained model to use on the encoder side; choose a name from Huggingface models',
    )

    parser.add_argument(
        '--num_workers', type=int, default=0, help='Number of processes to use for data loading (0 means no multiprocessing)'
    )

    parser.add_argument(
        '--rnn_dimension', default=None, type=int, help='output dimensions for RNN layers (for TransformerLSTM)'
    )
    parser.add_argument('--rnn_layers', default=1, type=int, help='number of layers for RNN modules ')
    parser.add_argument(
        '--rnn_zero_state',
        default='average',
        choices=['zero', 'average', 'cls'],
        help='how to construct RNN zero state (for TransformerLSTM)',
    )

    parser.add_argument(
        '--trainable_decoder_embeddings', default=50, type=int, help='size of decoder embedding (for TransformerLSTM)'
    )
    parser.add_argument('--dropout_ratio', default=0.2, type=float, help='dropout for the model (for TransformerLSTM)')

    parser.add_argument('--override_context', type=str, default=None, help='Override the context for all tasks')
    parser.add_argument('--override_question', type=str, default=None, help='Override the question for all tasks')
    # TODO for backward compatibility only. Remove after no old model (including paraphraser) is in use.
    parser.add_argument(
        '--no_separator',
        action='store_true',
        help='By default, we add a model-specific separator token between question and context when concatenating them. This argument disables that.',
    )
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
        '--model_parallel',
        action='store_true',
        help='Use model parallelization by spliting model weights across available gpus',
    )
    parser.add_argument(
        '--mp_device_ratio',
        default=None,
        nargs='+',
        type=int,
        help='Provide the distribution ratio of model layers across gpus when using model_parallel'
        'e.g. "1 2 2 2" first device recieves half number of layers compared to other devices'
        'default is None meaning we distribute evenly on all available gpus',
    )

    parser.add_argument('--warmup', default=40, type=int, help='warmup for learning rate. setting it to 1 disables warmup.')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping')
    parser.add_argument(
        '--beta0',
        default=0.9,
        type=float,
        help='alternative momentum for Adam (only when not using transformer scheduler), and RAdam',
    )
    parser.add_argument(
        '--optimizer',
        default='adam',
        choices=['adam', 'adamw', 'adafactor', 'radam', 'sgd'],
        type=str,
        help='optimizer to use',
    )
    parser.add_argument(
        '--lr_schedule',
        type=str,
        default='transformer',
        choices=['transformer', 'constant', 'linear', 'sgd', 'cosine'],
        help='The learning rate strategy. All of them can be used with or without warmup.',
    )
    parser.add_argument(
        '--lr_multiply',
        default=0.01,
        type=float,
        help='Multiplier for the `transformer` learning rate scheduler, constant value for `constant` and maximum value for `linear` and `cosine` schedulers.',
    )
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight L2 regularization')
    parser.add_argument(
        '-gas',
        '--gradient_accumulation_steps',
        default=1,
        type=int,
        help='Number of accumulation steps. Useful to effectively get larger batch sizes.',
    )

    # Loss Truncation; introduced in https://arxiv.org/abs/2004.14589
    parser.add_argument(
        '--dropper_ratio',
        type=float,
        default=0.0,
        help='Ratio of dropped examples in the "Loss Truncation" algorithm. 0 disables truncation.',
    )
    parser.add_argument(
        '--dropper_min_count',
        type=int,
        default=10000,
        help='Number of examples to see in the "Loss Truncation" algorithm before starting to drop high-loss examples.',
    )
    # Label smoothing; see https://arxiv.org/abs/1906.02629 for detailed analysis on its effect on neural network calibration
    parser.add_argument(
        '--label_smoothing',
        type=float,
        default=0.0,
        help='A number in [0, 1] to be used for label smoothing. 0 disables smoothing.',
    )

    parser.add_argument(
        '--load',
        default=None,
        type=str,
        help='path to checkpoint to load model from inside --args.save, usually set to best.pth',
    )
    parser.add_argument('--resume', action='store_true', help='whether to resume training with past optimizers')

    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used for training')

    parser.add_argument(
        '--no_commit',
        action='store_false',
        dest='commit',
        help='do not track the git commit associated with this training run',
    )
    parser.add_argument(
        '--exist_ok', action='store_true', help='Ok if the save directory already exists, i.e. overwrite is ok'
    )

    parser.add_argument(
        '--no_fast_tokenizer', action='store_true', help='Ignore all conditions and use slow version of huggingface tokenizer'
    )
    parser.add_argument(
        '--force_fast_tokenizer',
        action='store_true',
        help='Ignore all conditions and use fast version of huggingface tokenizer',
    )

    parser.add_argument('--skip_cache', action='store_true', help='whether to use existing cached splits or generate new ones')
    parser.add_argument(
        '--cache_input_data', action='store_true', help='Cache examples from input data for faster subsequent trainings'
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

    # NED args
    parser.add_argument('--do_ned', action='store_true', help='Collect and use entity features during training')
    parser.add_argument(
        '--min_entity_len',
        type=int,
        default=1,
        help='Minimum token-length of entities in ngram-based lookup for naive NED approach (does not apply to Bootleg)',
    )
    parser.add_argument(
        '--max_entity_len',
        type=int,
        default=4,
        help='Maximum token-length of entities in ngram-based lookup for naive NED approach (does not apply to Bootleg)',
    )
    parser.add_argument(
        '--database_dir',
        type=str,
        default='database/',
        help='Database folder containing all relevant files (e.g. alias2qids, pretrained models for bootleg)',
    )

    parser.add_argument(
        '--bootleg_output_dir',
        type=str,
        default='results_temp',
        help='Path to folder where bootleg prepped files should be saved',
    )
    parser.add_argument('--bootleg_model', type=str, default='bootleg_uncased_mini', help='Bootleg model to use')
    parser.add_argument(
        '--bootleg_prob_threshold',
        type=float,
        default=0.3,
        help='Probability threshold for accepting a candidate for a mention',
    )
    parser.add_argument(
        '--ned_normalize_types',
        default='off',
        choices=['off', 'soft', 'strict'],
        help='Normalize types. soft: attempt to map; if unsuccessful use original. strict: attempt to map; if unsuccessful drop the type.',
    )

    parser.add_argument(
        '--entity_type_agg_method',
        choices=['average', 'weighted'],
        default='average',
        help='Method used to aggregate several type embeddings for a single mention',
    )
    parser.add_argument(
        "--entity_word_embeds_dropout",
        default=0.0,
        type=float,
        help='Dropout entity word embeddings with this probability when encoding inputs',
    )

    parser.add_argument(
        "--add_entities_to_text",
        default='off',
        choices=['off', 'insert', 'append'],
        help='Method for adding entities to input text in text-based NER approach',
    )

    parser.add_argument(
        "--entity_attributes",
        nargs='+',
        default=['type_id'],
        help='Process only these entity attributes for adding them to text. Options are type_id, type_prob, and qid',
    )

    parser.add_argument(
        "--almond_type_mapping_path",
        default=None,
        type=str,
        help='If provided, will override the usual almond type mapping in data_utils/database_file/'
        'Path should be relative to --root',
    )

    parser.add_argument("--ned_dump_entity_type_pairs", action='store_true', help='Dump entity type pairs')
    parser.add_argument(
        '--ned_retrieve_method',
        default='bootleg',
        choices=['naive', 'entity-oracle', 'type-oracle', 'entity-type-oracle', 'bootleg'],
        type=str,
        help='how to retrieve types for entities',
    )

    parser.add_argument('--ned_domains', nargs='+', default=[], help='Domains used for almond dataset; e.g. music, books, ...')

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

    # token classification task args
    parser.add_argument('--num_labels', type=int, help='num_labels for classification tasks')
    parser.add_argument('--crossner_domains', nargs='+', type=str, help='domains to use for CrossNER task')
    parser.add_argument(
        '--hf_test_overfit',
        action='store_true',
        help='Debugging flag for hf datasets where validation will be performed on train set',
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
        for t in args.train_task_names:
            if t not in args.val_task_names:
                args.val_task_names.append(t)
    if 'imdb' in args.val_task_names:
        args.val_task_names.remove('imdb')

    args.timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime('%D-%H:%M:%S %Z')

    # TODO relax the following assertions by dropping samples from batches in Iterator
    if args.sentence_batching and args.train_batch_tokens[0] % len(args.train_src_languages.split('+')) != 0:
        raise ValueError(
            'Your train_batch_size should be divisible by number of train_src_languages when using sentence batching.'
        )
    if args.sentence_batching and args.val_batch_size[0] % len(args.eval_src_languages.split('+')) != 0:
        raise ValueError(
            'Your val_batch_size should be divisible by number of eval_src_languages when using sentence batching.'
        )

    if len(args.train_task_names) > 1:
        if args.train_iterations is None:
            args.train_iterations = [1]
        if len(args.train_iterations) < len(args.train_task_names):
            args.train_iterations = len(args.train_task_names) * args.train_iterations
        if len(args.train_batch_tokens) < len(args.train_task_names):

            args.train_batch_tokens = len(args.train_task_names) * args.train_batch_tokens

    # postprocess arguments
    if args.commit:
        args.commit = get_commit()
    else:
        args.commit = ''

    if have_multilingual(args.train_task_names) and (args.train_src_languages is None or args.eval_src_languages is None):
        raise ValueError('You have to define training and evaluation languages when you have a multilingual task')

    args.log_dir = args.save
    args.dist_sync_file = os.path.join(args.log_dir, 'distributed_sync_file')

    for x in ['data', 'save', 'log_dir', 'dist_sync_file']:
        setattr(args, x, os.path.join(args.root, getattr(args, x)))

    args.max_features_size = args.max_types_per_qid * args.max_qids_per_entity

    # tasks with the same name share the same task object
    train_tasks_dict = get_tasks(args.train_task_names, args)
    args.train_tasks = list(train_tasks_dict.values())
    val_task_dict = get_tasks(args.val_task_names, args, available_tasks=train_tasks_dict)
    args.val_tasks = list(val_task_dict.values())

    save_args(args)

    return args


def post_parse_train_specific(args):
    if len(args.val_batch_size) < len(args.val_task_names):
        args.val_batch_size = len(args.val_task_names) * args.val_batch_size

    if args.no_fast_tokenizer and args.force_fast_tokenizer:
        raise ValueError('Both no_fast_tokenizer and force_fast_tokenizer flags are on')

    # TODO relax this assertion by allowing training on multiple languages
    if 'mbart' in args.pretrained_model:
        if len(args.train_src_languages.split('+')) != 1 or set(args.train_src_languages.split('+')) != set(
            args.eval_src_languages.split('+')
        ):
            raise ValueError('For now we only support single language training and evaluation with mbart models')

    if args.model_parallel:
        if args.model == 'TransformerLSTM':
            raise ValueError('Model parallel is not supported for TransformerLSTM models')
        elif args.model == 'TransformerSeq2Seq' and args.pretrained_model not in MODEL_PARALLEL_SUPPORTED_MODELS:
            raise ValueError('Only the following models have model_parallel support: ', MODEL_PARALLEL_SUPPORTED_MODELS)

    if args.mp_device_ratio is not None:
        if len(args.mp_device_ratio) != len(args.devices):
            raise ValueError('When using model_parallel number of provided devices must match the number of mp_device_ratio')

    if args.warmup < 1:
        raise ValueError('Warmup should be a positive integer.')

    if args.use_encoder_loss and not (args.sentence_batching and len(args.train_src_languages.split('+')) > 1):
        raise ValueError('To use encoder loss you must use sentence batching and use more than one language during training.')

    if args.preprocess_special_tokens and args.model == 'TransformerLSTM':
        raise ValueError('Preprocessing special tokens should not be used for TransformerLSTM models')

    if args.model == 'TransformerLSTM' and 'uncased' in args.pretrained_model:
        raise ValueError(
            'You should use the cased version of provided model when not preprocessing special tokens.'
            ' Otherwise the program (answer) will not be tokenized properly'
        )

    if args.override_valid_metrics:
        assert len(args.override_valid_metrics) == len(args.train_tasks) == len(args.val_tasks)
        for train_task, val_task, metrics in zip(args.train_tasks, args.val_tasks, args.override_valid_metrics):
            train_task.metrics = metrics
            val_task.metrics = metrics

    args.log_dir = args.save
    if args.tensorboard_dir is None:
        args.tensorboard_dir = args.log_dir

    for x in ['embeddings']:
        setattr(args, x, os.path.join(args.root, getattr(args, x)))

    save_args(args, force_overwrite=True)

    args = check_and_update_generation_args(args)
    return args
