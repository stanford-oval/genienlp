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
import re
import sys

import numpy as np
import torch

from .data_utils.example import NumericalizedExamples
from .data_utils.iterator import LengthSortedIterator
from .tasks.generic_dataset import all_tokens_fn, default_batch_fn, input_tokens_fn

logger = logging.getLogger(__name__)

ENTITY_MATCH_REGEX = re.compile('^([A-Z].*)_[0-9]+$')
QUOTED_MATCH_REGEX = re.compile(' " (.*?) " ')
NUMBER_MATCH_REGEX = re.compile('^([0-9]+\.?[0-9]?)$')


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


def find_span(haystack, needle):
    for i in range(len(haystack) - len(needle) + 1):
        found = True
        for j in range(len(needle)):
            if haystack[i + j] != needle[j]:
                found = False
                break
        if found:
            return i
    return None


def make_data_loader(
    dataset, numericalizer, batch_size, return_original_order=False,
):
    all_features = NumericalizedExamples.from_examples(dataset, numericalizer)

    answer_lengths = [ex.answer.length for ex in all_features]

    min_batch_length = 1

    max_output_length = numericalizer.args.max_output_length

    if min_batch_length > batch_size:
        raise ValueError(
            f'The minimum batch length in your dataset is {min_batch_length} but your batch size is {batch_size}.'
            f' Thus no examples will be processed. Consider increasing batch_size'
        )
    if np.max(answer_lengths) > max_output_length:
        raise ValueError(
            f'The maximum output length in your dataset is {np.max(answer_lengths)} but you have set --max_output_length to {max_output_length}.'
            f' Consider increasing that'
        )

    sampler = LengthSortedIterator(
        all_features,
        batch_size=batch_size,
    )
    # get the sorted data_source
    all_f = sampler.data_source
    data_loader = torch.utils.data.DataLoader(
        all_f,
        batch_sampler=sampler,
        collate_fn=lambda batches: NumericalizedExamples.collate_batches(batches, numericalizer),
        num_workers=0,
    )

    if return_original_order:
        return data_loader, sampler.original_order
    else:
        return data_loader


def load_config_file_to_args(args):
    with open(os.path.join(args.path, 'config.json')) as config_file:
        config = json.load(config_file)

    retrieve = [
        'model',
        'preprocess_special_tokens',
        'override_valid_metrics',
    ]

    # train and predict scripts have these arguments in common. We use the values from train only if they are not provided in predict.
    # NOTE: do not set default values for these arguments in predict cause the defaults will always override training arguments
    overwrite = [
        'model',
        'val_batch_size',
        'top_p',
        'top_k',
        'repetition_penalty',
        'temperature',
        'max_output_length',
        'e2e_dialogue_valid_subtasks',
        'e2e_dialogue_valid_submetrics',
    ]
    for o in overwrite:
        if o not in args or getattr(args, o) is None:
            retrieve.append(o)

    # these are true/ false arguments
    overwrite_actions = [
        'e2e_dialogue_evaluation',
    ]
    for o in overwrite_actions:
        # if argument is True in predict overwrite train; if False retrieve from train
        if not getattr(args, o, False):
            retrieve.append(o)

    for r in retrieve:
        if r in config:
            setattr(args, r, config[r])
        elif r in ('top_p', 'repetition_penalty'):
            setattr(args, r, [1])
        else:
            # use default value
            setattr(args, r, None)

    if args.e2e_dialogue_valid_subtasks is None:
        setattr(args, 'e2e_dialogue_valid_subtasks', ['dst', 'api', 'da', 'rg'])
    if args.e2e_dialogue_valid_submetrics is None:
        setattr(args, 'e2e_dialogue_valid_submetrics', ['dst_em', 'em', 'da_em', 'casedbleu'])

    args.best_checkpoint = os.path.join(args.path, args.checkpoint_name)


def replace_capturing_group(input, re_pattern, replacement):
    # replace first captured group in the input with replacement using regex re_pattern
    if re_pattern.search(input):
        whole_match = re_pattern.search(input).group(0).strip()
        captured_match = re_pattern.search(input).group(1).strip()
        new_whole_match = whole_match.replace(captured_match, replacement)
        new_input = re.sub(re_pattern, new_whole_match, input)
    else:
        new_input = input
    return new_input


def print_results(results, num_print):
    print()

    values = list(results.values())
    num_examples = len(values[0])

    # examples are sorted by length
    # to get good diversity, get half of examples from second quartile
    start = int(num_examples / 4)
    end = start + int(num_print / 2)
    first_list = [val[start:end] for val in values]

    # and the other half from fourth quartile
    start = int(3 * num_examples / 4)
    end = start + num_print - int(num_print / 2)
    second_list = [val[start:end] for val in values]

    # join examples
    processed_values = [first + second for first, second in zip(first_list, second_list)]

    for ex_idx in range(len(processed_values[0])):
        for key_idx, key in enumerate(results.keys()):
            value = processed_values[key_idx][ex_idx]
            v = value[0] if isinstance(value, list) else value
            key_width = max(len(key) for key in results)
            print(f'{key:>{key_width}}: {repr(v)}')
        print()
    sys.stdout.flush()
