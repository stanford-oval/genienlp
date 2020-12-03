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
import numpy as np
import torch

from .data_utils.example import NumericalizedExamples
from .data_utils.numericalizer.sequential_field import SequentialField
from .data_utils.iterator import LengthSortedIterator
from .data_utils.progbar import prange

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


multiwoz_domain_names = ['Attraction', 'Hotel', 'Restaurant', 'Taxi', 'Train']
multiwoz_action_names = ['make_booking', 'make_reservation']

def multiwoz_specific_preprocess(thingtalk: str):
    thingtalk = thingtalk.strip()
    thingtalk, quote_values = remove_thingtalk_quotes(thingtalk)
    if quote_values is None:
        quote_values = []
    thingtalk = ' ' + thingtalk + ' '
    thingtalk = thingtalk.replace('@org.thingpedia.dialogue.transaction.', '')
    thingtalk = thingtalk.replace('$dialogue', '')
    thingtalk = thingtalk.replace('^^uk.ac.cam.multiwoz.', '')
    thingtalk = thingtalk.replace('param:', '')
    thingtalk = thingtalk.replace('@uk.ac.cam.multiwoz.', '')
    thingtalk = thingtalk.replace('enum:', '')
    thingtalk = thingtalk.replace('GENERIC_ENTITY_uk.ac.cam.multiwoz.', '')
    thingtalk = thingtalk.replace('QUOTED_', '')
    thingtalk = thingtalk.replace('#[', '[')
    thingtalk = thingtalk.replace('now => ;', '')
    thingtalk = thingtalk.replace('_', ' ')
    for a, b in [(d+'.make', d+' make') for d in multiwoz_domain_names]:
        thingtalk = thingtalk.replace(' '+a+' ', ' '+b+' ')

    # put the strings in quotes back
    for v in quote_values:
        thingtalk = thingtalk.replace('""', '" ' + v + ' "', 1)

    thingtalk = thingtalk.strip()
    return thingtalk

def multiwoz_specific_postprocess(thingtalk: str):
    thingtalk = thingtalk.strip()

    # replace string in quotes so that they don't accidentally change
    thingtalk, quote_values = remove_thingtalk_quotes(thingtalk)
    if quote_values is None:
        # The ThingTalk is not syntactically correct
        quote_values = []
    thingtalk = ' ' + thingtalk + ' '
    thingtalk = re.sub('(\S),', '\\1 ,', thingtalk)

    for a, b in [
        # recover parameter names that have underscore
        ('price range', 'price_range'),
        ('entrance fee', 'entrance_fee'),
        ('reference number', 'reference_number'),
        ('contact number', 'contact_number'),
        ('arrive by', 'arrive_by'),
        ('arrive at', 'arrive_at'),
        ('book day', 'book_day'),
        ('book stay', 'book_stay'),
        ('book people', 'book_people'),
        ('book time', 'book_time'),
        ('leave at', 'leave_at'),
        ('price stay', 'price_stay'),

        # recover enums that have underscore
        ('guest house', 'guest_house'),

        # recover system acts with underscore
        ('sys learn more what', 'sys_learn_more_what'),
        ('sys learn more', 'sys_learn_more'),
        ('sys goodbye', 'sys_goodbye'),
        ('sys greet', 'sys_greet'),
        ('sys success', 'sys_success'),
        ('sys invalid confirm', 'sys_invalid_confirm'),
        ('sys invalid propose action', 'sys_invalid_propose_action'),
        ('sys invalid action error question', 'sys_invalid_action_error_question'),
        ('sys invalid action success', 'sys_invalid_action_success'),
        ('sys invalid error question', 'sys_invalid_error_question'), 
        ('sys invalid', 'sys_invalid'),
        ('sys action success', 'sys_action_success'),
        ('sys action error question', 'sys_action_error_question'),
        ('sys action error', 'sys_action_error'),
        ('sys slot fill', 'sys_slot_fill'),
        ('sys propose refined query', 'sys_propose_refined_query'),
        ('sys recommend one slot fill', 'sys_recommend_one_slot_fill'), 
        ('sys recommend one', 'sys_recommend_one'),
        ('sys recommend two', 'sys_recommend_two'),
        ('sys recommend three', 'sys_recommend_three'),
        ('sys recommend four', 'sys_recommend_four'),
        ('sys recommend five', 'sys_recommend_five'),
        ('sys recommend seven', 'sys_recommend_seven'),
        ('sys recommend eight', 'sys_recommend_eight'),
        ('sys recommend nine', 'sys_recommend_nine'),
        ('sys recommend ten', 'sys_recommend_ten'),
        ('sys recommend eleven', 'sys_recommend_eleven'),
        ('sys recommend twelve', 'sys_recommend_twelve'),
        ('sys recommend thirteen', 'sys_recommend_thirteen'),
        ('sys recommend seventeen', 'sys_recommend_seventeen'),
        ('sys recommend twentyone', 'sys_recommend_twentyone'),
        ('sys execute', 'sys_execute'),
        ('sys generic search question', 'sys_generic_search_question'),
        ('sys generic error', 'sys_generic_error'),
        ('sys anything else', 'sys_anything_else'),
        ('sys search question', 'sys_search_question'),
        ('sys empty search question', 'sys_empty_search_question'),
        ('sys empty search', 'sys_empty_search'),
        ('sys action confirm', 'sys_action_confirm'),
        ('sys proposed refined query', 'sys_proposed_refined_query'),

        # recover user acts with underscore
        ('action question', 'action_question'),
        ('ask recommendation', 'ask_recommendation'),
        ('learn more', 'learn_more'),
        ('ask recommend', 'ask_recommend'),
        ('ask recommend', 'ask_recommend'),

        # recover actions with underscore
        ('make booking', 'make_booking'),
        ('make reservation', 'make_reservation'),

        ('in array~', 'in_array~'),
        ('in array', 'in_array'),
        ('[ confirm', '#[ confirm'),
    ] +\
        [('STRING '+str(i), 'QUOTED_STRING_'+str(i)) for i in range(0, 25)] +\
        [('TIME '+str(i), 'TIME_'+str(i)) for i in range(0, 25)] +\
        [('NUMBER '+str(i), 'NUMBER_'+str(i)) for i in range(0, 25)] +\
        [(d + ':' + d + ' ' + str(i), 'GENERIC_ENTITY_uk.ac.cam.multiwoz.' + d + ':' + d + '_' + str(i)) for i in range(0, 25) for d in multiwoz_domain_names] +\
            [(d + ':' + d, '^^uk.ac.cam.multiwoz.' + d + ':' + d) for d in multiwoz_domain_names]:
        thingtalk = thingtalk.replace(' ' + a + ' ', ' ' + b + ' ')

    # replace 'hotel' wherever it is an enum
    for a, b in [('type == hotel', 'type == enum:hotel'), ('[ hotel', '[ enum:hotel')]:
        thingtalk = thingtalk.replace(' ' + a + ' ', ' ' + b + ' ')

    # Add the prefix for system and user acts
    for a in ['sys_goodbye', 'sys_invalid', 'sys_greet', 'sys_execute', 'sys_success', 'sys_invalid_propose_action', 'sys_invalid_action_success', 'sys_invalid_confirm', 'sys_invalid_action_error_question', 'sys_recommend_seven', 'sys_recommend_eight', 'sys_recommend_nine', 'sys_recommend_ten', 'sys_recommend_eleven', 'sys_recommend_twelve', 'sys_recommend_thirteen', 'sys_invalid_error_question', 'sys_proposed_refined_query', 'sys_generic_error', 'sys_learn_more_what', 'sys_learn_more', 'sys_generic_search_question', 'sys_recommend_one_slot_fill', 'sys_recommend_one', 'sys_recommend_two', 'sys_recommend_three', 'sys_recommend_four', 'sys_recommend_five', 'sys_recommend_seventeen', 'sys_recommend_twentyone', 'sys_anything_else', 'sys_empty_search_question', 'sys_action_success', 'sys_empty_search', 'sys_propose_refined_query', 'sys_anything_else', 'sys_search_question', 'sys_action_error_question', 'sys_action_error', 'sys_slot_fill', 'sys_action_confirm',
              'execute',  'ask_recommendation', 'cancel', 'end', 'invalid', 'greet', 'goodbye', 'success', 'ask_recommend', 'action_question', 'insist', 'learn_more']:
        thingtalk = thingtalk.replace(' ' + a + ' ', ' $dialogue @org.thingpedia.dialogue.transaction.' + a + ' ')

    # First, connect domain name and action name with a dot to get e.g. Hotel.Hotel and Hotel.make_booking
    for a, b in [(d + ' ' + action, d + '.' + action) for d in multiwoz_domain_names for action in multiwoz_action_names]:
        thingtalk = thingtalk.replace(' ' + a + ' ', ' ' + b + ' ')
    # Second, add the prefix
    for a in [d+'.'+d for d in multiwoz_domain_names] + [d+'.'+action for d in multiwoz_domain_names for action in multiwoz_action_names]:
        thingtalk = thingtalk.replace(' ' + a + ' ', ' @uk.ac.cam.multiwoz.' + a + ' ')

    # Prepend enum: to enums
    for a in ['centre', 'west', 'east', 'north', 'south',
              'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
              'expensive', 'cheap', 'moderate', 'free',
              'guest_house',
              'proposed']:
        thingtalk = thingtalk.replace(' ' + a + ' ', ' enum:' + a + ' ')

    # Prepend param: to parameter names. Note that any remaining 'hotel' is a parameter name
    for a in ['area', 'price_range', 'address', 'parking', 'internet', 'type', 'postcode', 'entrance_fee',
              'phone', 'id', 'stars', 'book_day', 'book_people', 'book_stay', 'hotel', 'reference_number',
              'book_time', 'food', 'restaurant', 'leave_at', 'destination', 'arrival', 'departure', 'arrive_by', 'arrive_at', 'car',
              'day', 'train', 'price', 'duration', 'openhours', 'contact_number', 'people', 'time', 'price_stay']:
        thingtalk = thingtalk.replace(' ' + a + ' ', ' param:' + a + ' ')

    # put the strings in quotes back
    for v in quote_values:
        thingtalk = thingtalk.replace('""', '" ' + v + ' "', 1)
    thingtalk = thingtalk.strip()
    return thingtalk

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


def make_data_loader(dataset, numericalizer, batch_size, device=None, paired=False, max_pairs=None, train=False,
                     append_question_to_context_too=False, override_question=None, override_context=None, return_original_order=False):
    
    all_features = NumericalizedExamples.from_examples(dataset, numericalizer, device=device,
                                  paired=paired and train, max_pairs=max_pairs, groups=dataset.groups,
                                  append_question_to_context_too=append_question_to_context_too,
                                  override_question=override_question, override_context=override_context)

    all_f = []
    for i in prange(len(all_features.example_id), desc='Converting dataset to features'):
        all_f.append(NumericalizedExamples(example_id=[all_features.example_id[i]],
                            context=SequentialField(value=all_features.context.value[i], length=all_features.context.length[i], limited=all_features.context.limited[i]),
                            question=SequentialField(value=all_features.question.value[i], length=all_features.question.length[i], limited=all_features.question.limited[i]),
                            answer=SequentialField(value=all_features.answer.value[i], length=all_features.answer.length[i], limited=all_features.answer.limited[i]),
                            decoder_vocab=all_features.decoder_vocab, device=device, padding_function=numericalizer.pad))
    
    del all_features
    sampler = LengthSortedIterator(all_f, batch_size=batch_size, sort=True, shuffle_and_repeat=train, sort_key_fn=dataset.sort_key_fn, batch_size_fn=dataset.batch_size_fn, groups=dataset.groups)
    # get the sorted data_source
    all_f = sampler.data_source
    data_loader = torch.utils.data.DataLoader(all_f, batch_sampler=sampler, collate_fn=NumericalizedExamples.collate_batches, num_workers=0)
    
    if return_original_order:
        return data_loader, sampler.original_order
    else:
        return data_loader


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
                    'pretrain_context', 'pretrain_mlm_probability', 'force_subword_tokenize',
                    'append_question_to_context_too', 'almond_preprocess_context', 'almond_dataset_specific_preprocess', 'almond_lang_as_question',
                    'override_question', 'override_context', 'almond_has_multiple_programs']

        # train and predict scripts have these arguments in common. We use the values from train only if they are not provided in predict
        if 'num_beams' in config and not isinstance(config['num_beams'], list):
            # num_beams used to be an integer in previous versions of the code
            config['num_beams'] = [config['num_beams']]
        overwrite = ['val_batch_size', 'num_beams', 'num_outputs', 'no_repeat_ngram_size', 'top_p', 'top_k', 'repetition_penalty', 'temperature', 'reduce_metrics']
        for o in overwrite:
            if o not in args or getattr(args, o) is None:
                retrieve.append(o)

        for r in retrieve:
            if r in config:
                setattr(args, r, config[r])
            # These are for backward compatibility with models that were trained before we added these arguments
            elif r == 'almond_has_multiple_programs':
                setattr(args, r, False)
            elif r == 'almond_lang_as_question':
                setattr(args, r, False)
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
            elif r in ('append_question_to_context_too', 'almond_preprocess_context'):
                setattr(args, r, False)
            elif r == 'almond_dataset_specific_preprocess':
                setattr(args, r, 'none')
            elif r == 'num_beams':
                setattr(args, r, [1])
            elif r == 'num_outputs':
                setattr(args, r, [1])
            elif r == 'no_repeat_ngram_size':
                setattr(args, r, [0])
            elif r == 'top_p':
                setattr(args, r, [1.0])
            elif r == 'top_k':
                setattr(args, r, [0])
            elif r == 'repetition_penalty':
                setattr(args, r, [1.0])
            elif r == 'temperature':
                setattr(args, r, [0.0])
            elif r == 'reduce_metrics':
                setattr(args, r, 'max')
            else:
                setattr(args, r, None)
        args.dropout_ratio = 0.0

    args.best_checkpoint = os.path.join(args.path, args.checkpoint_name)
