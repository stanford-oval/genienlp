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
from json.decoder import JSONDecodeError
import logging
import os
import shutil
import random
import time
import re
from typing import List
import numpy as np
import torch
from transformers.models.mbart.tokenization_mbart import FAIRSEQ_LANGUAGE_CODES
from torch.functional import Tensor

from .data_utils.example import NumericalizedExamples
from .data_utils.iterator import LengthSortedIterator

logger = logging.getLogger(__name__)

ENTITY_MATCH_REGEX = re.compile('^([A-Z].*)_[0-9]+$')


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
    
    def forward(self, s: str):
        reverse_map = []
        matches = re.finditer(self.pattern, s)
        if matches is None:
            return s, reverse_map
        for match in matches:
            occurrence = match.group(0)
            parameter = match.group(1)
            replacement = self.forward_func(parameter)
            s = s.replace(occurrence, replacement)
            reverse_map.append((self, occurrence))
        return s, reverse_map

    def backward(self, s: str, occurrence: str):
        match = re.match(self.pattern, occurrence)
        parameter = match.group(1)
        if self.backward_func is None:
            list_of_strings_to_match = [self.forward_func(parameter)]
        else:
            list_of_strings_to_match = sorted(self.backward_func(parameter), key=lambda x:len(x), reverse=True)
        for string_to_match in list_of_strings_to_match:
            l = [' '+string_to_match+' ', string_to_match+' ', ' '+string_to_match]
            o = [' '+occurrence+' ', occurrence+' ', ' '+occurrence]
            new_s = s
            for i in range(len(l)):
                new_s = re.sub(l[i], o[i], s, flags=re.IGNORECASE)
                if s != new_s:
                    break
            if s != new_s:
                s = new_s
                break
            
        return s


class ConfidenceOutput:
    """
    Contains all necessary features that are useful for calculating confidence of a single generated output
    """

    def __init__(self, logits: List[Tensor], gold_answer: Tensor, prediction: Tensor):
        """
        Inputs:
            gold_answer: includes BOS and EOS tokens, but no PAD tokens
            prediction: includes BOS and EOS tokens, but no PAD tokens
        """
        self.logits = torch.stack(logits, dim=0)

        self.logit_mean = self.logits.mean(dim=0)
        self.logit_variance = self.logits.var(dim=0)
        self.logit_cv = torch.sqrt(self.logit_variance) / self.logit_mean # coefficient of variation
        self.logit_cv[self.logit_cv != self.logit_cv] = 0 # set NANs to 0

        self.first_mistake = ConfidenceOutput.find_first_mistake(gold_answer, prediction)

    @staticmethod
    def find_first_mistake(gold_answer: Tensor, prediction: Tensor):
        """
        Inputs:
            gold_answer: includes BOS and EOS tokens, but no PAD tokens
            prediction: includes BOS and EOS tokens, but no PAD tokens
        Returns:
            The index of the first token where `gold_answer` and `prediction` differ, or -1 if they are equal. Ignores BOS, so the minimum possible value is .
        """
        # print('gold_answer = ', gold_answer)
        # print('prediction = ', prediction)
        # skip BOS
        gold_answer = gold_answer[1:]
        prediction = prediction[1:]

        for i in range(min(len(gold_answer), len(prediction))):
            if gold_answer[i] != prediction[i]:
                return i
        if len(gold_answer) != len(prediction):
            # one is a strict prefix of the other
            return min(len(gold_answer), len(prediction))
        return -1

    def __repr__(self) -> str:
        return '<Confidence: logits=' + str(self.logits) \
                +', logit_mean='+ str(self.logit_mean) \
                + ', logit_variance='+ str(self.logit_variance) \
                + ', logit_cv='+ str(self.logit_cv) \
                + ', first_mistake='+ str(self.first_mistake) \
                + '>'


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


def find_span_type(program, begin_index, end_index):

    if begin_index > 1 and program[begin_index - 2] == 'location:':
        span_type = 'LOCATION'
    elif end_index == len(program) - 1 or not program[end_index + 1].startswith('^^'):
        span_type = 'QUOTED_STRING'
    else:
        if program[end_index + 1] == '^^tt:hashtag':
            span_type = 'HASHTAG'
        elif program[end_index + 1] == '^^tt:username':
            span_type = 'USERNAME'
        else:
            span_type = 'GENERIC_ENTITY_' + program[end_index + 1][2:]

        end_index += 1

    return span_type, end_index


def requote_program(program):
    
    program = program.split(' ')
    requoted = []

    in_string = False
    begin_index = 0
    i = 0
    while i < len(program):
        token = program[i]
        if token == '"':
            in_string = not in_string
            if in_string:
                begin_index = i + 1
            else:
                span_type, end_index = find_span_type(program, begin_index, i)
                requoted.append(span_type)
                i = end_index
           
        elif not in_string:
            entity_match = ENTITY_MATCH_REGEX.match(token)
            if entity_match is not None:
                requoted.append(entity_match[1])
            elif token != 'location:':
                requoted.append(token)
        
        i += 1
        
    return ' '.join(requoted)


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


def split_file_on_disk(file_path, num_splits, output_paths=None, delete=False):
    """
    """

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

    with open(file_path, 'r') as input_file:
        output_file_idx = 0
        for line in input_file:
            all_output_files[output_file_idx].write(line)
            output_file_idx = (output_file_idx + 1) % len(all_output_files)

    for f in all_output_files:
        f.close()
        
    if delete:
        os.remove(file_path)

    return all_output_paths


def combine_folders_on_disk(folder_path_prefix, num_files, line_group_size, delete=False):
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
        with open(new, 'w') as combined_file:
            if new.endswith('.json'):
                new_json = None
                for old in olds:
                    with open(old, 'r') as f:
                        if new_json is None:
                            try:
                                new_json = json.load(f)
                            except JSONDecodeError:
                                f.seek(0)
                                logger.info('Failed to read json file %s with content:\n %s', old, f.read())
                        else:
                            for k, v in json.load(f).items():
                                new_json[k] += v
                for k, v in new_json.items():
                    new_json[k] /= float(num_files)
                json.dump(new_json, combined_file)
            else:
                all_old_file_contents = []
                for old in olds:
                    with open(old, 'r') as f:
                        all_old_file_contents.append([line for line in f])
                old_file_idx = 0
                all_indices = [0] * len(all_old_file_contents)
                finished_reading = [False] * len(all_old_file_contents)
                while True:
                    if finished_reading[old_file_idx]:
                        old_file_idx = (old_file_idx + 1) % len(all_old_file_contents)
                        continue
                    for i in range(line_group_size):
                        line = all_old_file_contents[old_file_idx][all_indices[old_file_idx]]
                        combined_file.write(line)
                        all_indices[old_file_idx] += 1
                    if all_indices[old_file_idx] == len(all_old_file_contents[old_file_idx]):
                        finished_reading[old_file_idx] = True
                        if all(finished_reading):
                            break
                    old_file_idx = (old_file_idx + 1) % len(all_old_file_contents)
                
    if delete:
        for folder in folder_paths:
            shutil.rmtree(folder)


def combine_files_on_disk(file_path_prefix, num_files, line_group_size, delete=False):
    all_input_file_contents = []
    all_input_file_paths = []
    for i in range(num_files):
        input_file_path = get_part_path(file_path_prefix, i)
        all_input_file_paths.append(input_file_path)
        with open(input_file_path, 'r') as f:
            all_input_file_contents.append([line for line in f])
    
    all_indices = [0] * len(all_input_file_contents)
    finished_reading = [False] * len(all_input_file_contents)
    input_file_idx = 0
    with open(file_path_prefix, 'w') as combined_file:
        while True:
            if finished_reading[input_file_idx]:
                input_file_idx = (input_file_idx + 1) % len(all_input_file_contents)
                continue
            for i in range(line_group_size):
                line = all_input_file_contents[input_file_idx][all_indices[input_file_idx]]
                combined_file.write(line)
                all_indices[input_file_idx] += 1
            if all_indices[input_file_idx] == len(all_input_file_contents[input_file_idx]):
                finished_reading[input_file_idx] = True
                if all(finished_reading):
                    break
            input_file_idx = (input_file_idx + 1) % len(all_input_file_contents)

    if delete:
        for file_path in all_input_file_paths:
            os.remove(file_path)


def map_filter(callable, iterable):
    output = []
    for element in iterable:
        new_element = callable(element)
        if new_element is not None:
            output.append(new_element)
    return output


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
    #TODO is always called with name=False, so remove the if statement
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


def make_data_loader(dataset, numericalizer, batch_size, device=None, train=False, return_original_order=False):
    all_features = NumericalizedExamples.from_examples(dataset, numericalizer=numericalizer)

    context_lengths = [ex.context.length for ex in all_features]
    answer_lengths = [ex.answer.length for ex in all_features]

    logger.info(f'context lengths (min, mean, max): {np.min(context_lengths)}, {int(np.mean(context_lengths))}, {np.max(context_lengths)}')
    logger.info(f'answer lengths (min, mean, max): {np.min(answer_lengths)}, {int(np.mean(answer_lengths))}, {np.max(answer_lengths)}')
    
    sampler = LengthSortedIterator(all_features, batch_size=batch_size, sort=True, shuffle_and_repeat=train,
                                   sort_key_fn=dataset.sort_key_fn, batch_size_fn=dataset.batch_size_fn, groups=dataset.groups)
    # get the sorted data_source
    all_f = sampler.data_source
    data_loader = torch.utils.data.DataLoader(all_f, batch_sampler=sampler,
                                              collate_fn=lambda batches: NumericalizedExamples.collate_batches(batches, numericalizer, device),
                                              num_workers=0)
    
    if return_original_order:
        return data_loader, sampler.original_order
    else:
        return data_loader
    

def get_mbart_lang(orig_lang):
    for lang in FAIRSEQ_LANGUAGE_CODES:
        if lang.startswith(orig_lang):
            return lang

def have_multilingual(task_names):
    return any(['multilingual' in name for name in task_names])


def load_config_json(args):
    args.almond_type_embeddings = False
    with open(os.path.join(args.path, 'config.json')) as config_file:
        config = json.load(config_file)
        retrieve = ['model', 'pretrained_model', 'rnn_dimension', 'rnn_layers', 'rnn_zero_state',
                    'max_output_length', 'max_generative_vocab', 'lower', 'trainable_decoder_embeddings',
                    'override_context', 'override_question',
                    'almond_lang_as_question', 'almond_has_multiple_programs', 'almond_detokenize_sentence',
                    'preprocess_special_tokens', 'dropper_ratio', 'dropper_min_count']

        # train and predict scripts have these arguments in common. We use the values from train only if they are not provided in predict
        overwrite = ['val_batch_size', 'num_beams', 'num_beam_groups', 'diversity_penalty',
                     'num_outputs', 'no_repeat_ngram_size', 'top_p', 'top_k', 'repetition_penalty',
                     'temperature', 'reduce_metrics']
        for o in overwrite:
            if o not in args or getattr(args, o) is None:
                retrieve.append(o)

        for r in retrieve:
            if r in config:
                setattr(args, r, config[r])
            # These are for backward compatibility with models that were trained before we added these arguments
            elif r in ('preprocess_special_tokens'):
                setattr(args, r, False)
            elif r == 'num_beam_groups':
                setattr(args, r, [1])
            elif r == 'diversity_penalty':
                setattr(args, r, [0.0])
            elif r == 'dropper_ratio':
                setattr(args, r, 0.0)
            elif r == 'dropper_min_count':
                setattr(args, r, 10000)
            else:
                setattr(args, r, None)
        args.dropout_ratio = 0.0

    args.best_checkpoint = os.path.join(args.path, args.checkpoint_name)
