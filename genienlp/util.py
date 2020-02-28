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
import random
import time

import numpy as np
import torch

from .data_utils.example import Batch
from .data_utils.iterator import Iterator

logger = logging.getLogger(__name__)


def tokenizer(s):
    return s.split()

def detokenize(text):
    tokens = ["'d", "n't", "'ve", "'m", "'re", "'ll", ".", ",", "?", "'s", ")"]
    for t in tokens:
        text = text.replace(' ' + t, t)
    text = text.replace("( ", "(")
    return text

def get_number_of_lines(file_path):
    count = 0
    with open(file_path) as f:
        for line in f:
            count += 1
    return count

def get_file_part_path(file_path, part_idx):
    return file_path + '_part' + str(part_idx+1)

def split_file_on_disk(file_path, num_splits):
    """
    """
    number_of_lines = get_number_of_lines(file_path)

    all_output_paths = []
    all_output_files = []
    for i in range(num_splits):
        output_path = get_file_part_path(file_path, i)
        all_output_paths.append(output_path)
        all_output_files.append(open(output_path, 'w'))

    written_lines = 0
    with open(file_path, 'r') as input_file:
        output_file_idx = 0
        for line in input_file:
            all_output_files[output_file_idx].write(line)
            written_lines += 1
            if written_lines % (number_of_lines//num_splits) == 0:
                output_file_idx = min(output_file_idx + 1, len(all_output_files)-1)

    for f in all_output_files:
        f.close()

    return all_output_paths

def combine_files_on_disk(file_path_prefix, num_files):
    with open(file_path_prefix, 'w') as combined_file:
        for i in range(num_files):
            file_path = get_file_part_path(file_path_prefix, i)
            with open(file_path, 'r') as file:
                for line in file:
                    combined_file.write(line)

def map_filter(callable, iterable):
    output = []
    for element in iterable:
        new_element = callable(element)
        if new_element is not None:
            output.append(new_element)
    return output


def preprocess_examples(args, tasks, splits, logger=None, train=True):
    min_length = 1
    max_context_length = args.max_train_context_length if train else args.max_val_context_length
    is_too_long = lambda ex: (len(ex.answer) > args.max_answer_length or
                              len(ex.context) > max_context_length)
    is_too_short = lambda ex: (len(ex.answer) < min_length or
                               len(ex.context) < min_length)

    for task, s in zip(tasks, splits):
        if logger is not None:
            logger.info(f'{task.name} has {len(s.examples)} examples')

        l = len(s.examples)
        s.examples = map_filter(
            lambda ex: task.preprocess_example(ex, train=train, max_context_length=max_context_length),
            s.examples)

        if train:
            l = len(s.examples)
            s.examples = [ex for ex in s.examples if not is_too_long(ex)]
            if len(s.examples) < l:
                if logger is not None:
                    logger.info(f'Filtering out long {task.name} examples: {l} -> {len(s.examples)}')

            l = len(s.examples)
            s.examples = [ex for ex in s.examples if not is_too_short(ex)]
            if len(s.examples) < l:
                if logger is not None:
                    logger.info(f'Filtering out short {task.name} examples: {l} -> {len(s.examples)}')

        if logger is not None:
            context_lengths = [len(ex.context) for ex in s.examples]
            question_lengths = [len(ex.question) for ex in s.examples]
            answer_lengths = [len(ex.answer) for ex in s.examples]

            logger.info(
                f'{task.name} context lengths (min, mean, max): {np.min(context_lengths)}, {int(np.mean(context_lengths))}, {np.max(context_lengths)}')
            logger.info(
                f'{task.name} question lengths (min, mean, max): {np.min(question_lengths)}, {int(np.mean(question_lengths))}, {np.max(question_lengths)}')
            logger.info(
                f'{task.name} answer lengths (min, mean, max): {np.min(answer_lengths)}, {int(np.mean(answer_lengths))}, {np.max(answer_lengths)}')

        if logger is not None:
            logger.info('Tokenized examples:')
            for ex in s.examples[:10]:
                logger.info('Context: ' + ' '.join([token.strip() for token in ex.context]))
                logger.info('Question: ' + ' '.join([token.strip() for token in ex.question]))
                logger.info('Answer: ' + ' '.join([token.strip() for token in ex.answer]))


def init_devices(args, devices=None):
    if not torch.cuda.is_available():
        return [torch.device('cpu')]
    if not devices:
        return [torch.device('cuda:0')]
    return [torch.device(ordinal) for ordinal in devices]


def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def count_params(params):
    def mult(ps):
        r = 0
        for p in ps:
            this_r = 1
            for s in p.size():
                this_r *= s
            r += this_r
        return r

    return mult(params)


def get_trainable_params(model, name=False):
    if name:
        return list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    else:
        return list(filter(lambda p: p.requires_grad, model.parameters()))


def log_model_size(logger, model, model_name):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_param = count_params(params)
    logger.info(f'{model_name} has {num_param:,} parameters')


def elapsed_time(log):
    t = time.time() - log.start
    day = int(t // (24 * 3600))
    t = t % (24 * 3600)
    hour = int(t // 3600)
    t %= 3600
    minutes = int(t // 60)
    t %= 60
    seconds = int(t)
    return f'{day:02}:{hour:02}:{minutes:02}:{seconds:02}'


def batch_fn(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(len(new.context), 5 * len(new.answer), prev_max_len) * i


def make_data_loader(dataset, numericalizer, batch_size, device=None, train=False):
    iterator = Iterator(dataset, batch_size,
                        batch_size_fn=batch_fn if train else None,
                        shuffle=train,
                        repeat=train,
                        bucket_by_sort_key=train)
    return torch.utils.data.DataLoader(iterator, batch_size=None,
                                       collate_fn=lambda minibatch: Batch.from_examples(minibatch, numericalizer,
                                                                                        device=device))


def pad(x, new_channel, dim, val=None):
    if x.size(dim) > new_channel:
        x = x.narrow(dim, 0, new_channel)
    channels = x.size()
    assert (new_channel >= channels[dim])
    if new_channel == channels[dim]:
        return x
    size = list(channels)
    size[dim] = new_channel - size[dim]
    padding = x.new(*size).fill_(val)
    return torch.cat([x, padding], dim)


def load_config_json(args):
    args.almond_type_embeddings = False
    with open(os.path.join(args.path, 'config.json')) as config_file:
        config = json.load(config_file)
        retrieve = ['model', 'seq2seq_encoder', 'seq2seq_decoder', 'transformer_layers', 'rnn_layers', 'rnn_zero_state',
                    'transformer_hidden', 'dimension', 'rnn_dimension', 'load', 'max_val_context_length',
                    'val_batch_size', 'transformer_heads', 'max_output_length', 'max_generative_vocab', 'lower',
                    'encoder_embeddings', 'decoder_embeddings', 'trainable_decoder_embeddings',
                    'train_encoder_embeddings', 'locale', 'use_pretrained_bert']

        for r in retrieve:
            if r in config:
                setattr(args, r, config[r])
            elif r == 'locale':
                setattr(args, r, 'en')
            elif r == 'trainable_decoder_embedding':
                setattr(args, r, 0)
            elif r == 'train_encoder_embedding':
                setattr(args, r, False)
            elif r == 'rnn_dimension':
                setattr(args, r, args.dimension)
            elif r == 'rnn_zero_state':
                setattr(args, r, 'zero')
            elif r == 'use_pretrained_bert':
                setattr(args, r, True)
            else:
                setattr(args, r, None)
        args.dropout_ratio = 0.0

    args.best_checkpoint = os.path.join(args.path, args.checkpoint_name)
