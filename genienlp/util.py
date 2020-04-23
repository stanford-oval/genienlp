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
import shutil
import random
import time
import re
import numpy as np
import torch
import torch.nn.functional as F

from .data_utils.example import Batch
from .data_utils.iterator import Iterator

logger = logging.getLogger(__name__)


class SpecialTokenMap:
    def __init__(self, pattern, forward_func, backward_func=None):
        """
        Inputs:
            pattern: a regex pattern
            forward_func: a function with signature forward_func(str) -> str
            backward_func: a function with signature backward_func(str) -> list[str]
        """
        if isinstance(forward_func, list):
            self.forward_func = lambda x: forward_func[int(x)%len(forward_func)]
        else:
            self.forward_func = forward_func

        if isinstance(backward_func, list):
            self.backward_func = lambda x: backward_func[int(x)%len(backward_func)]
        else:
            self.backward_func = backward_func
    
        self.pattern = pattern
    
    def forwad(self, s: str):
        reverse_map = []
        matches = re.finditer(self.pattern, s)
        if matches is None:
            return s, reverse_map
        for match in matches:
            occurance = match.group(0)
            # print('occurance = ', occurance)
            parameter = match.group(1)
            replacement = self.forward_func(parameter)
            s = s.replace(occurance, replacement)
            reverse_map.append((self, occurance))
        return s, reverse_map

    def backward(self, s: str, occurance: str):
        match = re.match(self.pattern, occurance)
        parameter = match.group(1)
        if self.backward_func is None:
            list_of_strings_to_match = [self.forward_func(parameter)]
        else:
            list_of_strings_to_match = sorted(self.backward_func(parameter), key=lambda x:len(x), reverse=True)
        # print('list_of_strings_to_match = ', list_of_strings_to_match)
        for string_to_match in list_of_strings_to_match:
            l = [' '+string_to_match+' ', string_to_match+' ', ' '+string_to_match]
            o = [' '+occurance+' ', occurance+' ', ' '+occurance]
            new_s = s
            for i in range(len(l)):
                new_s = re.sub(l[i], o[i], s, flags=re.IGNORECASE)
                if s != new_s:
                    break
            if s != new_s:
                s = new_s
                break
            
        return s


def tokenizer(s):
    return s.split()

def mask_special_tokens(string: str):
    exceptions = [match.group(0) for match in re.finditer('[A-Za-z:_.]+_[0-9]+', string)]
    for e in exceptions:
        string = string.replace(e, '<temp>', 1)
    return string, exceptions

def unmask_special_tokens(string: str, exceptions: list):
    for e in exceptions:
        string = string.replace('<temp>', e, 1)
    return string


def detokenize(string: str):
    string, exceptions = mask_special_tokens(string)
    tokens = ["'d", "n't", "'ve", "'m", "'re", "'ll", ".", ",", "?", "!", "'s", ")", ":"]
    for t in tokens:
        string = string.replace(' ' + t, t)
    string = string.replace("( ", "(")
    string = string.replace('gon na', 'gonna')
    string = string.replace('wan na', 'wanna')
    string = unmask_special_tokens(string, exceptions)
    return string

def tokenize(string: str):
    string, exceptions = mask_special_tokens(string)
    tokens = ["'d", "n't", "'ve", "'m", "'re", "'ll", ".", ",", "?", "!", "'s", ")", ":"]
    for t in tokens:
        string = string.replace(t, ' ' + t)
    string = string.replace("(", "( ")
    string = string.replace('gonna', 'gon na')
    string = string.replace('wanna', 'wan na')
    string = re.sub('\s+', ' ', string)
    string = unmask_special_tokens(string, exceptions)
    return string.strip()

def lower_case(string):
    string, exceptions = mask_special_tokens(string)
    string = string.lower()
    string = unmask_special_tokens(string, exceptions)
    return string

def remove_thingtalk_quotes(thingtalk):
    quote_values = []
    while True:
        # print('before: ', thingtalk)
        l1 = thingtalk.find('"')
        if l1 < 0:
            break
        l2 = thingtalk.find('"', l1+1)
        if l2 < 0:
            # ThingTalk code is not syntactic
            return thingtalk, None
        quote_values.append(thingtalk[l1+1: l2].strip())
        thingtalk = thingtalk[:l1] + '<temp>' + thingtalk[l2+1:]
        # print('after: ', thingtalk)
    thingtalk = thingtalk.replace('<temp>', '""')
    return thingtalk, quote_values

def get_number_of_lines(file_path):
    count = 0
    with open(file_path) as f:
        for line in f:
            count += 1
    return count

def get_part_path(path, part_idx):
    if path.endswith(os.path.sep):
        has_separator = True
        path = path[:-1]
    else:
        has_separator = False
    return path + '_part' + str(part_idx+1) + (os.path.sep if has_separator else '')

def split_folder_on_disk(folder_path, num_splits):
    new_folder_paths = [get_part_path(folder_path, part_idx) for part_idx in range(num_splits)]
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            new_file_paths = [os.path.join(subdir.replace(folder_path, new_folder_paths[part_idx]), file) for part_idx in range(num_splits)]
            split_file_on_disk(os.path.join(subdir, file), num_splits, output_paths=new_file_paths)
    return new_folder_paths


def split_file_on_disk(file_path, num_splits, output_paths=None):
    """
    """
    number_of_lines = get_number_of_lines(file_path)

    all_output_paths = []
    all_output_files = []
    for part_idx in range(num_splits):
        if output_paths is None:
            output_path = get_part_path(file_path, part_idx)
        else:
            output_path = output_paths[part_idx]
        all_output_paths.append(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        all_output_files.append(open(output_path, 'w'))

    written_lines = 0
    with open(file_path, 'r') as input_file:
        output_file_idx = 0
        for line in input_file:
            all_output_files[output_file_idx].write(line)
            written_lines += 1
            if number_of_lines >= num_splits and written_lines % (number_of_lines//num_splits) == 0:
                output_file_idx = min(output_file_idx + 1, len(all_output_files)-1)

    for f in all_output_files:
        f.close()

    return all_output_paths

def combine_folders_on_disk(folder_path_prefix, num_files, delete=False):
    folder_paths = [get_part_path(folder_path_prefix, part_idx) for part_idx in range(num_files)]
    new_to_olds_map = {}
    for i in range(num_files):
        for subdir, dirs, files in os.walk(folder_paths[i]):
            for file in files:
                new_file_path = os.path.join(subdir.replace(folder_paths[i], folder_path_prefix), file)
                if new_file_path not in new_to_olds_map:
                    new_to_olds_map[new_file_path] = []    
                new_to_olds_map[new_file_path].append(os.path.join(subdir, file))
    
    for new, olds in new_to_olds_map.items():
        os.makedirs(os.path.dirname(new), exist_ok=True)
        new_json = None
        with open(new, 'w') as combined_file:
            for i in range(num_files):
                old_file_path = olds[i]
                with open(old_file_path, 'r') as old_file:
                    if new.endswith('.json'):
                        if new_json is None:
                            new_json = json.load(old_file)
                        else:
                            for k, v in json.load(old_file).items():
                                new_json[k] += v
                    else:
                        for line in old_file:
                            combined_file.write(line)
            if new.endswith('.json'):
                for k, v in new_json.items():
                    new_json[k] /= float(num_files)
                json.dump(new_json, combined_file)
    if delete:
        for folder in folder_paths:
            shutil.rmtree(folder)

def combine_files_on_disk(file_path_prefix, num_files, delete=False):
    with open(file_path_prefix, 'w') as combined_file:
        for i in range(num_files):
            file_path = get_part_path(file_path_prefix, i)
            with open(file_path, 'r') as file:
                for line in file:
                    combined_file.write(line)
            if delete:
                os.remove(file_path)

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf'), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_k < 0: keeps all tokens but the ones with top |k| probability
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    sign = 1
    if top_k < 0:
        top_k = logits.size(-1) + top_k
        sign = -1
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = sign*logits < torch.topk(sign*logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

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
        return [torch.device('cuda:'+str(i)) for i in range(torch.cuda.device_count())]
    return [torch.device(ordinal) for ordinal in devices]


def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_trainable_params(model, name=False):
    if name:
        return list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    else:
        return list(filter(lambda p: p.requires_grad, model.parameters()))


def log_model_size(logger, model, model_name):
    num_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])
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


def make_data_loader(dataset, numericalizer, batch_size, device=None, train=False, valid=False):
    
    iterator = Iterator(dataset, batch_size,
                        shuffle=train,
                        repeat=train,
                        use_data_batch_fn=train or valid,
                        bucket_by_sort_key=train or valid)
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


def have_multilingual(task_names):
    return any(['multilingual' in name for name in task_names])


def load_config_json(args):
    args.almond_type_embeddings = False
    with open(os.path.join(args.path, 'config.json')) as config_file:
        config = json.load(config_file)
        retrieve = ['model', 'seq2seq_encoder', 'seq2seq_decoder', 'transformer_layers', 'rnn_layers', 'rnn_zero_state',
                    'transformer_hidden', 'dimension', 'rnn_dimension', 'load', 'max_val_context_length',
                    'transformer_heads', 'max_output_length', 'max_generative_vocab', 'lower',
                    'encoder_embeddings', 'context_embeddings', 'question_embeddings', 'decoder_embeddings',
                    'trainable_decoder_embeddings', 'trainable_encoder_embeddings', 'train_encoder_embeddings',
                    'train_context_embeddings', 'train_question_embeddings', 'locale', 'use_pretrained_bert',
                    'train_context_embeddings_after', 'train_question_embeddings_after',
                    'pretrain_context', 'pretrain_mlm_probability', 'force_subword_tokenize']

        # train and predict scripts have these arguments in common. We use the values from train only if they are not provided in predict
        overwrite = ['val_batch_size', 'num_beams']
        for o in overwrite:
            if getattr(args, o) is None:
                retrieve.append(o)

        for r in retrieve:
            if r in config:
                setattr(args, r, config[r])
            # backward compatibility with models that were trained before we added these arguments
            elif r == 'locale':
                setattr(args, r, 'en')
            elif r in ('trainable_decoder_embedding', 'trainable_encoder_embeddings', 'pretrain_context',
                       'train_context_embeddings_after', 'train_question_embeddings_after'):
                setattr(args, r, 0)
            elif r == 'pretrain_mlm_probability':
                setattr(args, r, 0.15)
            elif r == 'context_embeddings':
                if args.seq2seq_encoder == 'Coattention':
                    setattr(args, r, '')
                else:
                    setattr(args, r, args.encoder_embeddings)
            elif r == 'question_embeddings':
                setattr(args, r, args.encoder_embeddings)
            elif r == 'train_encoder_embeddings':
                setattr(args, r, False)
            elif r == 'train_context_embeddings':
                if args.seq2seq_encoder == 'Coattention':
                    setattr(args, r, False)
                else:
                    setattr(args, r, args.train_encoder_embeddings)
            elif r == 'train_question_embeddings':
                setattr(args, r, args.train_encoder_embeddings)
            elif r == 'rnn_dimension':
                setattr(args, r, args.dimension)
            elif r == 'rnn_zero_state':
                setattr(args, r, 'zero')
            elif r == 'use_pretrained_bert':
                setattr(args, r, True)
            elif r == 'num_beams':
                setattr(args, r, 1)
            else:
                setattr(args, r, None)
        args.dropout_ratio = 0.0

    args.best_checkpoint = os.path.join(args.path, args.checkpoint_name)
