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


import time
import os
import torch
import random
import numpy as np
import ujson as json
import logging


logger = logging.getLogger(__name__)

def tokenizer(s):
    return s.split()

def get_context_question(ex, context, question, field):
    return ex.context_special + ex.context + ex.question_special + ex.question


def preprocess_examples(args, tasks, splits, field, logger=None, train=True):
    min_length = 1
    max_context_length = args.max_train_context_length if train else args.max_val_context_length
    is_too_long = lambda ex: (len(ex.answer) > args.max_answer_length or
        len(ex.context)>max_context_length)
    is_too_short = lambda ex: (len(ex.answer) < min_length or
        len(ex.context)<min_length)

    for task, s in zip(tasks, splits):
        if logger is not None:
            logger.info(f'{task.name} has {len(s.examples)} examples')

        l = len(s.examples)
        s.examples = list(filter(lambda ex: task.preprocess_example(ex, train=train, max_context_length=max_context_length), s.examples))

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

            logger.info(f'{task.name} context lengths (min, mean, max): {np.min(context_lengths)}, {int(np.mean(context_lengths))}, {np.max(context_lengths)}')
            logger.info(f'{task.name} question lengths (min, mean, max): {np.min(question_lengths)}, {int(np.mean(question_lengths))}, {np.max(question_lengths)}')
            logger.info(f'{task.name} answer lengths (min, mean, max): {np.min(answer_lengths)}, {int(np.mean(answer_lengths))}, {np.max(answer_lengths)}')

        for x in s.examples:
            x.context_question = get_context_question(x, x.context, x.question, field)

        if logger is not None:
            logger.info('Tokenized examples:')
            for ex in s.examples[:10]:
                logger.info('Context: ' + ' '.join([token.strip() for token in ex.context]))
                logger.info('Question: ' + ' '.join([token.strip() for token in ex.question]))
                logger.info(' '.join([token.strip() for token in ex.context_question]))
                logger.info('Answer: ' + ' '.join([token.strip() for token in ex.answer]))



def set_seed(args, rank=None):
    if not torch.cuda.is_available():
        ordinal = -1
    elif rank is None and len(args.devices) > 0:
        ordinal = args.devices[0]
    else:
        ordinal = args.devices[rank] 
    device = torch.device(f'cuda:{ordinal}' if ordinal > -1 else 'cpu')
    # device = torch.device(f'cuda:{ordinal}' if ordinal > -1 else 'cpu')
    logger.debug(f'device: {device}')
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with torch.cuda.device(ordinal):
        torch.cuda.manual_seed(args.seed)
    return device


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
    return max(len(new.context), 5*len(new.answer), prev_max_len) * i


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
        retrieve = ['model', 'transformer_layers', 'rnn_layers', 'transformer_hidden',
                    'dimension', 'load', 'max_val_context_length', 'val_batch_size',
                    'transformer_heads', 'max_output_length', 'max_generative_vocab',
                    'lower', 'cove', 'intermediate_cove', 'elmo', 'glove_and_char',
                    'use_maxmargin_loss', 'small_glove', 'almond_type_embeddings',
                    'almond_grammar', 'trainable_decoder_embedding', 'glove_decoder',
                    'pretrained_decoder_lm']

        for r in retrieve:
            if r in config:
                setattr(args, r, config[r])
            elif r in ('cove', 'intermediate_cove', 'use_maxmargin_loss', 'small_glove',
                       'almond_type_embbedings'):
                setattr(args, r, False)
            elif 'elmo' in r:
                setattr(args, r, [-1])
            elif r in ('glove_decoder', 'glove_and_char'):
                setattr(args, r, True)
            elif r == 'trainable_decoder_embedding':
                setattr(args, r, 0)
            else:
                setattr(args, r, None)
        args.dropout_ratio = 0.0

    args.best_checkpoint = os.path.join(args.path, args.checkpoint_name)
