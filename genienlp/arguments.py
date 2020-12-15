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
from .util import have_multilingual
from .paraphrase.transformers_utils import BART_MODEL_LIST, MBART_MODEL_LIST, MT5_MODEL_LIST


logger = logging.getLogger(__name__)


def get_commit():
    directory = os.path.dirname(__file__)
    return subprocess.Popen("cd {} && git log | head -n 1".format(directory), shell=True,
                            stdout=subprocess.PIPE).stdout.read().split()[1].decode()


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def parse_argv(parser):
    parser.add_argument('--root', default='.', type=str,
                        help='root directory for data, results, embeddings, code, etc.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--save', required=True, type=str, help='where to save results.')
    parser.add_argument('--embeddings', default='.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--cache', default='.cache/', type=str, help='where to save cached files')

    parser.add_argument('--train_languages', type=str,
                        help='used to specify dataset languages used during training for multilingual tasks'
                             'multiple languages for each task should be concatenated with +')
    parser.add_argument('--eval_languages', type=str,
                        help='used to specify dataset languages used during validation for multilingual tasks'
                             'multiple languages for each task should be concatenated with +')

    parser.add_argument('--train_tasks', nargs='+', type=str, dest='train_task_names', help='tasks to use for training',
                        required=True)
    parser.add_argument('--train_iterations', nargs='+', type=int, help='number of iterations to focus on each task')
    #TODO rename to train_batch_size; keeping it for now for backward compatibility
    parser.add_argument('--train_batch_tokens', nargs='+', default=[4000], type=int,
                        help='Number of tokens to use for dynamic batching, corresponding to tasks in train tasks.'
                        'If sentence_batching is used, this will be interpreted as number of examples.')
    parser.add_argument('--jump_start', default=0, type=int, help='number of iterations to give jump started tasks')
    parser.add_argument('--n_jump_start', default=0, type=int, help='how many tasks to jump start (presented in order)')
    parser.add_argument('--num_print', default=15, type=int,
                        help='how many validation examples with greedy output to print to std out')

    parser.add_argument('--no_tensorboard', action='store_false', dest='tensorboard',
                        help='Turn off tensorboard logging')
    parser.add_argument('--tensorboard_dir', default=None,
                        help='Directory where to save Tensorboard logs (defaults to --save)')
    parser.add_argument('--max_to_keep', default=5, type=int, help='number of checkpoints to keep')
    parser.add_argument('--log_every', default=int(1e2), type=int, help='how often to log results in # of iterations')
    parser.add_argument('--save_every', default=int(1e3), type=int,
                        help='how often to save a checkpoint in # of iterations')

    parser.add_argument('--val_tasks', nargs='+', type=str, dest='val_task_names',
                        help='tasks to collect evaluation metrics for')
    parser.add_argument('--val_every', default=int(1e3), type=int,
                        help='how often to run validation in # of iterations')
    parser.add_argument('--val_no_filter', action='store_false', dest='val_filter',
                        help='whether to allow filtering on the validation sets')
    parser.add_argument('--val_batch_size', nargs='+', default=[4000], type=int,
                        help='Number of tokens in each batch for validation, corresponding to tasks in --val_tasks')
    
    parser.add_argument('--paired', action='store_true',
                        help='Pair related examples before numericalizing the input (e.g. training with synthetic and paraphrase '
                             'sentence pairs for almond task)')
    parser.add_argument('--max_pairs', type=int, default=1000000,
                        help='Maximum number of pairs to make for each example group')
    
    parser.add_argument('--sentence_batching', action='store_true',
                        help='Batch same sentences together (used for multilingual tasks)')
    parser.add_argument('--use_encoder_loss', action='store_true', help='Force encoded values for sentences in different languages to be the same')
    parser.add_argument('--encoder_loss_type', type=str, default='mean', choices=['mean', 'sum'],
                        help='Function to calculate encoder_loss_type from the context rnn hidden states')
    parser.add_argument('--encoder_loss_weight', type=float, default=0.1,
                        help='multiplicative constant choosing the weight of encoder_loss in total loss')
    parser.add_argument('--eval_set_name', type=str, help='Evaluation dataset name to use during training')

    parser.add_argument('--max_output_length', default=100, type=int, help='maximum output length for generation')
    parser.add_argument('--max_generative_vocab', default=50000, type=int,
                        help='max vocabulary for the generative softmax')
    parser.add_argument('--max_train_context_length', default=500, type=int,
                        help='maximum length of the contexts during training')
    parser.add_argument('--max_val_context_length', default=500, type=int,
                        help='maximum length of the contexts during validation')
    parser.add_argument('--max_answer_length', default=50, type=int,
                        help='maximum length of answers during training and validation')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the datasets')
    parser.add_argument('--preserve_case', action='store_false', dest='lower',
                        help='whether to preserve casing for all text')
    parser.add_argument("--reduce_metrics", type=str, default='max', choices=['max'], help='How to calculate the metric when there are multiple outputs per input.')

    # These are generation hyperparameters. Each one can be a list of values in which case, we generate `num_outputs` outputs for each set of hyperparameters.
    parser.add_argument("--num_outputs", type=int, nargs='+', default=[1], help='number of sequences to output per input')
    parser.add_argument("--temperature", type=float, nargs='+', default=[0.0],
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, nargs='+', default=[1.0],
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[1.0], help='1.0 disables top-p filtering')
    parser.add_argument("--num_beams", type=int, nargs='+', default=[1], help='1 disables beam seach')
    parser.add_argument("--no_repeat_ngram_size", type=int, nargs='+', default=[0], help='ngrams of this size cannot be repeated in the output. 0 disables it.')

    parser.add_argument("--almond_has_multiple_programs", action='store_true', help='Indicate if almond dataset has multiple programs for each sentence')

    parser.add_argument('--model', type=str, choices=['Seq2Seq', 'Bart', 'MT5', 'MBart'], default='Seq2Seq', help='which model to import')
    parser.add_argument('--seq2seq_encoder', type=str, choices=['MQANEncoder', 'BiLSTM', 'Identity', 'Coattention'],
                        default='MQANEncoder', help='which encoder to use for the Seq2Seq model')
    parser.add_argument('--seq2seq_decoder', type=str, choices=['MQANDecoder'] + BART_MODEL_LIST + MBART_MODEL_LIST + MT5_MODEL_LIST, default='MQANDecoder',
                        help='which decoder to use for the Seq2Seq model')
    parser.add_argument('--dimension', default=200, type=int, help='output dimensions for all layers')
    parser.add_argument('--rnn_dimension', default=None, type=int, help='output dimensions for RNN layers')
    parser.add_argument('--rnn_layers', default=1, type=int, help='number of layers for RNN modules')
    parser.add_argument('--rnn_zero_state', default='zero', choices=['zero', 'average', 'cls'],
                        help='how to construct RNN zero state (for Identity encoder)')
    parser.add_argument('--transformer_layers', default=2, type=int, help='number of layers for transformer modules')
    parser.add_argument('--transformer_hidden', default=150, type=int, help='hidden size of the transformer modules')
    parser.add_argument('--transformer_heads', default=3, type=int, help='number of heads for transformer modules')
    parser.add_argument('--dropout_ratio', default=0.2, type=float, help='dropout for the model')

    parser.add_argument('--encoder_embeddings', default=None,
                        help='which word embedding to use on the encoder side; use `glove+char`, a bert-* model for pretrained BERT; or a xlm-roberta* model for Multi-lingual RoBERTa; '
                             'multiple embeddings can be concatenated with +; use @0, @1 to specify untied copies')
    parser.add_argument('--context_embeddings', default=None,
                        help='which word embedding to use for the context; use a bert-* pretrained model for BERT; '
                             'multiple embeddings can be concatenated with +; use @0, @1 to specify untied copies')
    parser.add_argument('--question_embeddings', default=None,
                        help='which word embedding to use for the question; use a bert-* pretrained model for BERT; '
                             'multiple embeddings can be concatenated with +; use @0, @1 to specify untied copies')
    parser.add_argument('--train_encoder_embeddings', action='store_true', default=False,
                        help='back propagate into pretrained encoder embedding (recommended for BERT and XLM-RoBERTa)')
    parser.add_argument('--train_context_embeddings', action='store_true', default=None,
                        help='back propagate into pretrained context embedding (recommended for BERT and XLM-RoBERTa)')
    parser.add_argument('--train_context_embeddings_after', type=int, default=0,
                        help='back propagate into pretrained context embedding after the given iteration (default: '
                             'immediately)')
    parser.add_argument('--train_question_embeddings', action='store_true', default=None,
                        help='back propagate into pretrained question embedding (recommended for BERT)')
    parser.add_argument('--train_question_embeddings_after', type=int, default=0,
                        help='back propagate into pretrained context embedding after the given iteration (default: '
                             'immediately)')
    parser.add_argument('--decoder_embeddings', default='glove+char',
                        help='which pretrained word embedding to use on the decoder side')
    parser.add_argument('--trainable_encoder_embeddings', default=0, type=int,
                        help='size of trainable portion of encoder embedding (only for Coattention encoder)')
    parser.add_argument('--trainable_decoder_embeddings', default=0, type=int,
                        help='size of trainable portion of decoder embedding (0 or omit to disable)')
    parser.add_argument('--pretrain_context', default=0, type=int,
                        help='number of pretraining steps for the context encoder')
    parser.add_argument('--pretrain_mlm_probability', default=0.15, type=int,
                        help='probability of replacing a token with mask for MLM pretraining')
    parser.add_argument('--force_subword_tokenize', action='store_true', default=False,
                        help='force subword tokenization of code tokens too')
    parser.add_argument('--append_question_to_context_too', action='store_true', default=False,
                        help='')
    parser.add_argument('--override_question', type=str, default=None, help='Override the question for all tasks')
    parser.add_argument('--override_context', type=str, default=None, help='Override the context for all tasks')
    parser.add_argument('--almond_preprocess_context', action='store_true', default=False, help='')
    parser.add_argument('--almond_dataset_specific_preprocess', type=str, default='none', choices=['none', 'multiwoz'],
                        help='Applies dataset-sepcific preprocessing to context and answer fields, and postprocesses the model outputs back to the original form.')
    parser.add_argument('--almond_lang_as_question', action='store_true',
                        help='if true will use "Translate from ${language} to ThingTalk" for question')

    parser.add_argument('--warmup', default=800, type=int, help='warmup for learning rate')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping')
    parser.add_argument('--beta0', default=0.9, type=float,
                        help='alternative momentum for Adam (only when not using transformer_lr)')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'radam'], type=str,
                        help='optimizer to use')
    parser.add_argument('--no_transformer_lr', action='store_false', dest='transformer_lr',
                        help='turns off the transformer learning rate strategy')
    parser.add_argument('--transformer_lr_multiply', default=1.0, type=float,
                        help='multiplier for transformer learning rate (if using Adam)')
    parser.add_argument('--lr_rate', default=0.001, type=float, help='fixed learning rate (if not using warmup)')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight L2 regularization')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='Number of accumulation steps. Useful to effectively get larger batch sizes.')

    parser.add_argument('--load', default=None, type=str, help='path to checkpoint to load model from inside --args.save, usually set to best.pth')
    parser.add_argument('--resume', action='store_true', help='whether to resume training with past optimizers')

    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--devices', default=[0], nargs='+', type=int,
                        help='a list of devices that can be used for training')

    parser.add_argument('--no_commit', action='store_false', dest='commit',
                        help='do not track the git commit associated with this training run')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Ok if the save directory already exists, i.e. overwrite is ok')

    parser.add_argument('--skip_cache', action='store_true',
                        help='whether to use existing cached splits or generate new ones')
    parser.add_argument('--cache_input_data', action='store_true',
                        help='Cache examples from input data for faster subsequent trainings')
    parser.add_argument('--use_curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--aux_dataset', default='', type=str,
                        help='path to auxiliary dataset (ignored if curriculum is not used)')
    parser.add_argument('--curriculum_max_frac', default=1.0, type=float,
                        help='max fraction of harder dataset to keep for curriculum')
    parser.add_argument('--curriculum_rate', default=0.1, type=float, help='growth rate for curriculum')
    parser.add_argument('--curriculum_strategy', default='linear', type=str, choices=['linear', 'exp'],
                        help='growth strategy for curriculum')


def post_parse(args):
    if args.val_task_names is None:
        args.val_task_names = []
        for t in args.train_task_names:
            if t not in args.val_task_names:
                args.val_task_names.append(t)
    if 'imdb' in args.val_task_names:
        args.val_task_names.remove('imdb')

    args.timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    
    def indices_of_multilingual(train_task_names):
        indices = []
        for i, task in enumerate(train_task_names):
            if 'multilingual' in task:
                indices.append(i)
        return indices
    
    #TODO relax the following assertions by dropping samples from batches in Iterator
    if args.sentence_batching and args.train_batch_tokens[0] % len(args.train_languages.split('+')) != 0:
        raise ValueError('Your train_batch_size should be divisible by number of train_languages when using sentence batching.')
    if args.sentence_batching and args.val_batch_size[0] % len(args.eval_languages.split('+')) != 0:
        raise ValueError('Your val_batch_size should be divisible by number of eval_languages when using sentence batching.')
    
    if args.warmup < 1:
        raise ValueError('Warmup should be a positive integer.')
    if args.use_encoder_loss and not (args.sentence_batching and len(args.train_languages.split('+')) > 1) :
        raise ValueError('To use encoder loss you must use sentence batching and use more than one language during training.')

    if args.override_context and args.append_question_to_context_too:
        raise ValueError('You cannot use append_question_to_context_too when overriding context')
    
    if args.paired and not args.sentence_batching:
        logger.warning('Paired training only works if sentence_batching is used as well.'
                        'Activating sentence_batching...')
        args.sentence_batching = True
        

    if len(args.train_task_names) > 1:
        if args.train_iterations is None:
            args.train_iterations = [1]
        if len(args.train_iterations) < len(args.train_task_names):
            args.train_iterations = len(args.train_task_names) * args.train_iterations
        if len(args.train_batch_tokens) < len(args.train_task_names):
            args.train_batch_tokens = len(args.train_task_names) * args.train_batch_tokens
    if args.sentence_batching:
        if args.paired:
            # TODO unify train_batch_tokens and train_batch_size
            train_batch_size = int(args.train_batch_tokens[0])
            num_train_langs = len(args.train_languages.split('+'))
            new_batch_size = int(train_batch_size *
                                 (1 + min(num_train_langs**2 - num_train_langs, args.max_pairs) / num_train_langs))
            logger.warning('Using paired example training will increase effective batch size from {} to {}'.
                            format(train_batch_size, new_batch_size))
        
    if len(args.val_batch_size) < len(args.val_task_names):
        args.val_batch_size = len(args.val_task_names) * args.val_batch_size

    # postprocess arguments
    if args.commit:
        args.commit = get_commit()
    else:
        args.commit = ''

    if args.rnn_dimension is None:
        args.rnn_dimension = args.dimension

    if args.context_embeddings is None:
        args.context_embeddings = args.encoder_embeddings
    if args.question_embeddings is None:
        args.question_embeddings = args.context_embeddings
    if args.train_context_embeddings is None:
        args.train_context_embeddings = args.train_encoder_embeddings
    if args.train_question_embeddings is None:
        args.train_question_embeddings = args.train_encoder_embeddings

    args.log_dir = args.save
    if args.tensorboard_dir is None:
        args.tensorboard_dir = args.log_dir
    args.dist_sync_file = os.path.join(args.log_dir, 'distributed_sync_file')
    
    if have_multilingual(args.train_task_names) and (args.train_languages is None or args.eval_languages is None):
        raise ValueError('You have to define training and evaluation languages when you have a multilingual task')
    
    for x in ['data', 'save', 'embeddings', 'log_dir', 'dist_sync_file']:
        setattr(args, x, os.path.join(args.root, getattr(args, x)))
    save_args(args)

    # create the task objects after we saved the configuration to the JSON file, because
    # tasks are not JSON serializable
    # tasks with the same name share the same task object
    train_tasks_dict = get_tasks(args.train_task_names, args)
    args.train_tasks = list(train_tasks_dict.values())
    val_task_dict = get_tasks(args.val_task_names, args, available_tasks=train_tasks_dict)
    args.val_tasks = list(val_task_dict.values())
    return args
