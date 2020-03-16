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

    parser.add_argument('--train_tasks', nargs='+', type=str, dest='train_task_names', help='tasks to use for training',
                        required=True)
    parser.add_argument('--train_iterations', nargs='+', type=int, help='number of iterations to focus on each task')
    parser.add_argument('--train_batch_tokens', nargs='+', default=[9000], type=int,
                        help='Number of tokens to use for dynamic batching, corresponding to tasks in train tasks')
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
    parser.add_argument('--val_batch_size', nargs='+', default=[256], type=int,
                        help='Batch size for validation corresponding to tasks in val tasks')

    parser.add_argument('--vocab_tasks', nargs='+', type=str, help='tasks to use in the construction of the vocabulary')
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
    parser.add_argument('--num_beams', type=int, default=1, help='number of beams to use for beam search')

    parser.add_argument('--model', type=str, choices=['Seq2Seq'], default='Seq2Seq', help='which model to import')
    parser.add_argument('--seq2seq_encoder', type=str, choices=['MQANEncoder', 'BiLSTM', 'Identity', 'Coattention'],
                        default='MQANEncoder', help='which encoder to use for the Seq2Seq model')
    parser.add_argument('--seq2seq_decoder', type=str, choices=['MQANDecoder'], default='MQANDecoder',
                        help='which decoder to use for the Seq2Seq model')
    parser.add_argument('--dimension', default=200, type=int, help='output dimensions for all layers')
    parser.add_argument('--rnn_dimension', default=None, type=int, help='output dimensions for RNN layers')
    parser.add_argument('--rnn_layers', default=1, type=int, help='number of layers for RNN modules')
    parser.add_argument('--rnn_zero_state', default='zero', choices=['zero', 'average'],
                        help='how to construct RNN zero state (for Identity encoder)')
    parser.add_argument('--transformer_layers', default=2, type=int, help='number of layers for transformer modules')
    parser.add_argument('--transformer_hidden', default=150, type=int, help='hidden size of the transformer modules')
    parser.add_argument('--transformer_heads', default=3, type=int, help='number of heads for transformer modules')
    parser.add_argument('--dropout_ratio', default=0.2, type=float, help='dropout for the model')

    parser.add_argument('--encoder_embeddings', default='glove+char',
                        help='which word embedding to use on the encoder side; use a bert-* pretrained model for BERT; or a xlm-roberta* model for Multi-lingual RoBERTa; '
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
    

    parser.add_argument('--load', default=None, type=str, help='path to checkpoint to load model from inside args.save')
    parser.add_argument('--resume', action='store_true', help='whether to resume training with past optimizers')

    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--devices', default=[0], nargs='+', type=int,
                        help='a list of devices that can be used for training')

    parser.add_argument('--no_commit', action='store_false', dest='commit',
                        help='do not track the git commit associated with this training run')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Ok if the save directory already exists, i.e. overwrite is ok')

    parser.add_argument('--skip_cache', action='store_true',
                        help='whether to use exisiting cached splits or generate new ones')
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

    if len(args.train_task_names) > 1:
        if args.train_iterations is None:
            args.train_iterations = [1]
        if len(args.train_iterations) < len(args.train_task_names):
            args.train_iterations = len(args.train_task_names) * args.train_iterations
        if len(args.train_batch_tokens) < len(args.train_task_names):
            args.train_batch_tokens = len(args.train_task_names) * args.train_batch_tokens
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

    for x in ['data', 'save', 'embeddings', 'log_dir', 'dist_sync_file']:
        setattr(args, x, os.path.join(args.root, getattr(args, x)))
    save_args(args)

    # create the task objects after we saved the configuration to the JSON file, because
    # tasks are not JSON serializable
    args.train_tasks = get_tasks(args.train_task_names, args)
    args.val_tasks = get_tasks(args.val_task_names, args)

    return args
