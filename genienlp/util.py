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
import shutil
import time
from json.decoder import JSONDecodeError
from typing import List, Optional

import numpy as np
import torch
import ujson
from torch.functional import Tensor
from transformers import MarianConfig, MBartConfig
from transformers.models.mbart50.tokenization_mbart50 import FAIRSEQ_LANGUAGE_CODES

from .data_utils.almond_utils import token_type_regex
from .data_utils.example import NumericalizedExamples
from .data_utils.iterator import LengthSortedIterator
from .model_utils.transformers_utils import MARIAN_GROUP_MEMBERS

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
            self.forward_func = lambda x: forward_func[int(x) % len(forward_func)]
        else:
            self.forward_func = forward_func

        if isinstance(backward_func, list):
            self.backward_func = lambda x: backward_func[int(x) % len(backward_func)]
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
            list_of_strings_to_match = sorted(self.backward_func(parameter), key=lambda x: len(x), reverse=True)
        for string_to_match in list_of_strings_to_match:
            l_ = [' ' + string_to_match + ' ', string_to_match + ' ', ' ' + string_to_match]
            o_ = [' ' + occurrence + ' ', occurrence + ' ', ' ' + occurrence]
            new_s = s
            for i in range(len(l_)):
                new_s = re.sub(l_[i], o_[i], s, flags=re.IGNORECASE)
                if s != new_s:
                    break
            if s != new_s:
                s = new_s
                break

        return s


class ConfidenceFeatures:
    """
    Contains all necessary features that are useful for calculating confidence of a single generated output
    """

    def __init__(
        self,
        drop_logits: List[Tensor],
        drop_probs: List[Tensor],
        gold_answer: Tensor,
        prediction: Tensor,
        nodrop_logits: Tensor,
        nodrop_probs: Tensor,
        nodrop_entropies: Tensor,
        context: Tensor,
        nodrop_top1_probs: Tensor = None,
        nodrop_top2_probs: Tensor = None,
        drop_top1_probs: List[Tensor] = None,
        drop_top2_probs: List[Tensor] = None,
    ):
        """
        Inputs:
            droplogits: logits after MC dropout
            gold_answer: includes BOS and EOS tokens, but no PAD tokens
            prediction: includes BOS and EOS tokens, but no PAD tokens
            nodrop_logits: logits for this prediction that are obtained WITHOUT activating model's dropout
            nodrop_top1_probs, nodrop_top2_probs: highest and second highest probabilities of the next token, given that the previous token was from `prediction`
        """

        # store the results of MC dropout if provided
        self.drop_logits = drop_logits
        self.drop_probs = drop_probs
        self.drop_top1_probs = drop_top1_probs
        self.drop_top2_probs = drop_top2_probs

        if drop_logits is not None:
            self.drop_logits = torch.stack(drop_logits, dim=0).cpu()
        if drop_probs is not None:
            self.drop_probs = torch.stack(drop_probs, dim=0).cpu()
        if drop_top1_probs is not None:
            self.drop_top1_probs = torch.stack(drop_top1_probs, dim=0).cpu()
        if drop_top2_probs is not None:
            self.drop_top2_probs = torch.stack(drop_top2_probs, dim=0).cpu()

        # store the results of non-dropout forward pass, if provided
        self.nodrop_logits = nodrop_logits
        self.nodrop_probs = nodrop_probs
        self.nodrop_entropies = nodrop_entropies
        self.nodrop_top1_probs = nodrop_top1_probs
        self.nodrop_top2_probs = nodrop_top2_probs

        if nodrop_logits is not None:
            self.nodrop_logits = nodrop_logits.cpu()
        if nodrop_probs is not None:
            self.nodrop_probs = nodrop_probs.cpu()
        if nodrop_entropies is not None:
            self.nodrop_entropies = nodrop_entropies.cpu()
        if nodrop_top1_probs is not None:
            self.nodrop_top1_probs = nodrop_top1_probs.cpu()
        if nodrop_top2_probs is not None:
            self.nodrop_top2_probs = nodrop_top2_probs.cpu()

        self.prediction = prediction
        self.gold_answer = gold_answer
        self.first_mistake = ConfidenceFeatures.find_first_mistake(gold_answer, prediction)
        self.label = self.first_mistake == -1
        self.context = context

    @property
    def mc_dropout_num(self):
        if self.drop_logits is None:
            return 0
        else:
            return self.drop_logits.shape[0]

    @staticmethod
    def find_first_mistake(gold_answer: Tensor, prediction: Tensor):
        """
        Inputs:
            gold_answer: includes BOS and EOS tokens, but no PAD tokens
            prediction: includes BOS and EOS tokens, but no PAD tokens
        Returns:
            The index of the first token where `gold_answer` and `prediction` differ, or -1 if they are equal. Ignores BOS, so the minimum possible value is 0.
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
        return (
            '<Confidence: drop_logits='
            + str(self.drop_logits)
            + ', drop_probs='
            + str(self.drop_probs)
            + ', first_mistake='
            + str(self.first_mistake)
            + ', nodrop_logits='
            + str(self.nodrop_logits)
            + ', nodrop_probs='
            + str(self.nodrop_probs)
            + ', nodrop_entropies='
            + str(self.nodrop_entropies)
            + ', context='
            + str(self.context)
            + ', label='
            + str(self.label)
            + '>'
        )


class GenerationOutput:
    """
    Contains all the information that the generation function may need to output
    """

    def __init__(
        self,
        loss: Optional[float] = None,
        example_ids: Optional[List] = None,
        predictions: Optional[List] = None,
        raw_predictions: Optional[List] = None,
        answers: Optional[List] = None,
        contexts: Optional[List] = None,
        confidence_features: Optional[List] = None,
        confidence_scores: Optional[List] = None,
    ):
        self.loss = loss
        self.example_ids = example_ids
        self.predictions = predictions
        self.raw_predictions = raw_predictions
        self.answers = answers
        self.contexts = contexts
        self.confidence_features = confidence_features
        self.confidence_scores = confidence_scores


def remove_thingtalk_quotes(thingtalk):
    quote_values = []
    while True:
        # print('before: ', thingtalk)
        l1 = thingtalk.find('"')
        if l1 < 0:
            break
        l2 = thingtalk.find('"', l1 + 1)
        if l2 < 0:
            # ThingTalk code is not syntactic
            return thingtalk, None
        quote_values.append(thingtalk[l1 + 1 : l2].strip())
        thingtalk = thingtalk[:l1] + '<temp>' + thingtalk[l2 + 1 :]
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
    tokens = ["'d", "n't", "'ve", "'m", "'re", "'ll", ".", ",", "?", "!", "'s", ")", ":", "-"]
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
    string = unmask_special_tokens(string, exceptions)
    string = re.sub('([A-Za-z:_.]+_[0-9]+)-', r'\1 - ', string)  # add space before and after hyphen, e.g. "NUMBER_0-hour"
    string = re.sub('\s+', ' ', string)  # remove duplicate spaces
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
    return path + '_part' + str(part_idx + 1) + (os.path.sep if has_separator else '')


def split_folder_on_disk(folder_path, num_splits):
    new_folder_paths = [get_part_path(folder_path, part_idx) for part_idx in range(num_splits)]
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            # ignore system files
            if file.startswith('.'):
                continue
            new_file_paths = [
                os.path.join(subdir.replace(folder_path, new_folder_paths[part_idx]), file) for part_idx in range(num_splits)
            ]
            split_file_on_disk(os.path.join(subdir, file), num_splits, output_paths=new_file_paths)
    return new_folder_paths


def split_file_on_disk(file_path, num_splits, output_paths=None, delete=False):
    """ """

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


def get_devices(devices=None):
    if not torch.cuda.is_available():
        return [torch.device('cpu')]
    if not devices:
        return [torch.device('cuda:' + str(i)) for i in range(torch.cuda.device_count())]
    return [torch.device(ordinal) for ordinal in devices]


def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_trainable_params(model, name=False):
    # TODO is always called with name=False, so remove the if statement
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
    all_features = NumericalizedExamples.from_examples(dataset, numericalizer)

    context_lengths = [ex.context.length for ex in all_features]
    answer_lengths = [ex.answer.length for ex in all_features]

    logger.info(
        f'context lengths (min, mean, max): {np.min(context_lengths)}, {int(np.mean(context_lengths))}, {np.max(context_lengths)}'
    )
    logger.info(
        f'answer lengths (min, mean, max): {np.min(answer_lengths)}, {int(np.mean(answer_lengths))}, {np.max(answer_lengths)}'
    )

    sampler = LengthSortedIterator(
        all_features,
        batch_size=batch_size,
        sort=True,
        shuffle_and_repeat=train,
        sort_key_fn=dataset.sort_key_fn,
        batch_size_fn=dataset.batch_size_fn,
        groups=dataset.groups,
    )
    # get the sorted data_source
    all_f = sampler.data_source
    data_loader = torch.utils.data.DataLoader(
        all_f,
        batch_sampler=sampler,
        collate_fn=lambda batches: NumericalizedExamples.collate_batches(batches, numericalizer, device),
        num_workers=0,
    )

    if return_original_order:
        return data_loader, sampler.original_order
    else:
        return data_loader


def ned_dump_entity_type_pairs(dataset, path, name, utterance_field):

    with open(os.path.join(path, f'{name}_labels.jsonl'), 'w') as fout:
        for ex in dataset.examples:
            text = ex.question if utterance_field == 'question' else ex.context

            span_beg = text.index('<e>')
            sentence = text[:span_beg].strip()
            entity_span = text[span_beg:].strip()
            entity_token_string = entity_span[len('<e>') : -len('</e>')].strip('; ')

            entities = []
            ent_types = []
            if entity_token_string:
                entity_token_pairs = entity_token_string.split(';')
                for str in entity_token_pairs:
                    entity, types = token_type_regex.match(str).groups()
                    types = types.split('|')
                    entities.append(entity.strip())
                    ent_types.append(types)

            fout.write(ujson.dumps({"sentence": sentence, "aliases": entities, "thingtalk_types": ent_types}) + '\n')


def merge_translated_sentences(
    example_ids, predictions, raw_predictions, answers, contexts, confidence_features, src_lang, tgt_lang
):
    new_example_ids = []
    new_predictions = []
    new_raw_predictions = []
    new_answers = []
    new_contexts = []
    new_confidence_features = []
    cur_pred, cur_raw_pred, cur_context, cur_answer = [], [], [], []
    i = 0
    src_concat_token = '' if src_lang in ['zh', 'ja', 'ko'] else ' '
    tgt_concat_token = '' if tgt_lang in ['zh', 'ja', 'ko'] else ' '
    while i < len(predictions):
        ex_id, pred, raw_pred, ans, ctxt, cf_feat = (
            example_ids[i],
            predictions[i],
            raw_predictions[i],
            answers[i],
            contexts[i],
            confidence_features[i],
        )
        if '@' in ex_id:
            id_, split_id = ex_id.rsplit('@', 1)
            cur_id = id_
            while id_ == cur_id:
                cur_pred.append(pred)
                cur_raw_pred.append(raw_pred)
                cur_context.append(ctxt)
                cur_answer.append(ans)
                i += 1
                if i < len(predictions):
                    ex_id, pred, raw_pred, ans, ctxt, cf_feat = (
                        example_ids[i],
                        predictions[i],
                        raw_predictions[i],
                        answers[i],
                        contexts[i],
                        confidence_features[i],
                    )
                    if '@' in ex_id:
                        id_, split_id = ex_id.rsplit('@', 1)
                    else:
                        id_ = ex_id
                else:
                    break

            new_example_ids.append(cur_id)
            new_predictions.append(
                [tgt_concat_token.join([cur_pred[j][0] for j in range(len(cur_pred))]) for i in range(len(cur_pred[0]))]
            )
            new_raw_predictions.append(
                [
                    tgt_concat_token.join([cur_raw_pred[j][0] for j in range(len(cur_raw_pred))])
                    for i in range(len(cur_raw_pred[0]))
                ]
            )
            new_contexts.append(src_concat_token.join(cur_context))
            new_answers.append(src_concat_token.join(cur_answer))
            new_confidence_features.append(cf_feat)

            # reset block
            cur_pred, cur_raw_pred, cur_context, cur_answer = [], [], [], []

        else:
            new_example_ids.append(ex_id)
            new_predictions.append(pred)
            new_raw_predictions.append(raw_pred)
            new_contexts.append(ctxt)
            new_answers.append(ans)
            new_confidence_features.append(cf_feat)
            i += 1

    return new_example_ids, new_predictions, new_raw_predictions, new_answers, new_contexts, new_confidence_features


def get_mbart_lang(orig_lang):
    for lang in FAIRSEQ_LANGUAGE_CODES:
        if lang.startswith(orig_lang):
            return lang


def adjust_language_code(config, pretrained_model, src_lang, tgt_lang):

    # adjust src and tgt languages for Marian models
    model_is_marian = isinstance(config, MarianConfig)

    if model_is_marian and pretrained_model.rsplit('-', 2)[1] in MARIAN_GROUP_MEMBERS:
        if src_lang not in MARIAN_GROUP_MEMBERS[pretrained_model.rsplit('-', 2)[1]]:
            if src_lang == 'pl':
                src_lang = 'pol'
            elif src_lang == 'fa':
                src_lang = 'pes'
            else:
                raise ValueError(
                    f'Source language "{src_lang}" is not in this Marian model group languages, please specify the correct source language.'
                )

    if model_is_marian and pretrained_model.rsplit('-', 1)[1] in MARIAN_GROUP_MEMBERS:
        if tgt_lang not in MARIAN_GROUP_MEMBERS[pretrained_model.rsplit('-', 1)[1]]:
            if tgt_lang == 'pl':
                tgt_lang = 'pol'
            elif tgt_lang == 'fa':
                tgt_lang = 'pes'
            else:
                raise ValueError(
                    f'Target language "{tgt_lang}" is not in this Marian model group languages, please specify the correct target language.'
                )

    if model_is_marian and pretrained_model.rsplit('-', 2)[1] not in MARIAN_GROUP_MEMBERS:
        # Source language should not be provided when using marian models with single language pairs
        # otherwise the translation outputs will be incorrect; hence we ignore the source language
        src_lang = None

    if model_is_marian and pretrained_model.rsplit('-', 1)[1] not in MARIAN_GROUP_MEMBERS:
        # Target language should not be provided when using marian models with single language pairs
        # otherwise the translation outputs will be incorrect; hence we ignore the target language
        tgt_lang = None

    # adjust src and tgt languages for Mbart models
    if isinstance(config, MBartConfig):
        src_lang = get_mbart_lang(src_lang)
        tgt_lang = get_mbart_lang(tgt_lang)

    return src_lang, tgt_lang


def have_multilingual(task_names):
    return any(['multilingual' in name for name in task_names])


def load_config_json(args):
    args.almond_type_embeddings = False
    with open(os.path.join(args.path, 'config.json')) as config_file:
        config = json.load(config_file)

        retrieve = [
            'model',
            'pretrained_model',
            'rnn_dimension',
            'rnn_layers',
            'rnn_zero_state',
            'max_generative_vocab',
            'lower',
            'trainable_decoder_embeddings',
            'override_context',
            'override_question',
            'almond_lang_as_question',
            'almond_has_multiple_programs',
            'almond_detokenize_sentence',
            'preprocess_special_tokens',
            'dropper_ratio',
            'dropper_min_count',
            'label_smoothing',
            'use_encoder_loss',
            'num_workers',
            'no_fast_tokenizer',
            'force_fast_tokenizer',
            'add_entities_to_text',
            'entity_attributes',
            'max_qids_per_entity',
            'max_types_per_qid',
            'do_ned',
            'database_type',
            'min_entity_len',
            'max_entity_len',
            'entity_type_agg_method',
            'entity_word_embeds_dropout',
            'num_db_types',
            'db_unk_id',
            'ned_retrieve_method',
            'ned_domains',
            'almond_type_mapping_path',
            'max_features_size',
            'bootleg_output_dir',
            'bootleg_model',
            'bootleg_prob_threshold',
            'ned_normalize_types',
            'att_pooling',
            'no_separator',
            'num_labels',
            'crossner_domains',
            'hf_test_overfit',
            'override_valid_metrics',
            'eval_src_languages',
            'eval_tgt_languages',
        ]

        # train and predict scripts have these arguments in common. We use the values from train only if they are not provided in predict
        overwrite = [
            'val_batch_size',
            'num_beams',
            'num_beam_groups',
            'diversity_penalty',
            'num_outputs',
            'no_repeat_ngram_size',
            'top_p',
            'top_k',
            'repetition_penalty',
            'temperature',
            'max_output_length',
            'reduce_metrics',
            'database_dir',
        ]
        # these are true/ false arguments
        overwrite_actions = ['do_alignment', 'align_preserve_input_quotation', 'align_remove_output_quotation']
        for o in overwrite:
            if o not in args or getattr(args, o) is None:
                retrieve.append(o)
        for o in overwrite_actions:
            if not getattr(args, o, False):
                retrieve.append(o)

        for r in retrieve:
            if r in config:
                setattr(args, r, config[r])
            # These are for backward compatibility with models that were trained before we added these arguments
            elif r in (
                'do_ned',
                'do_alignment',
                'align_preserve_input_quotation',
                'align_remove_output_quotation',
                'use_encoder_loss',
                'almond_has_multiple_programs',
                'almond_lang_as_question',
                'preprocess_special_tokens',
                'no_fast_tokenizer',
                'force_fast_tokenizer',
            ):
                setattr(args, r, False)
            elif r in ('ned_normalize_types'):
                setattr(args, r, 'off')
            elif r in ('num_db_types', 'db_unk_id', 'num_workers'):
                setattr(args, r, 0)
            elif r in ('entity_word_embeds_dropout'):
                setattr(args, r, 0.0)
            elif r in ('num_beams', 'num_outputs', 'top_p', 'repetition_penalty'):
                setattr(args, r, [1])
            elif r in ('no_repeat_ngram_size', 'top_k', 'temperature'):
                setattr(args, r, [0])
            elif r in ['override_valid_metrics']:
                setattr(args, r, [])
            elif r == 'database_type':
                setattr(args, r, 'json')
            elif r == 'att_pooling':
                setattr(args, r, 'max')
            elif r == 'min_entity_len':
                setattr(args, r, 2)
            elif r == 'max_entity_len':
                setattr(args, r, 4)
            elif r == 'ned_retrieve_method':
                setattr(args, r, 'naive')
            elif r == 'locale':
                setattr(args, r, 'en')
            elif r == 'num_beam_groups':
                setattr(args, r, [1])
            elif r == 'diversity_penalty':
                setattr(args, r, [0.0])
            elif r == 'dropper_ratio':
                setattr(args, r, 0.0)
            elif r == 'dropper_min_count':
                setattr(args, r, 10000)
            elif r == 'label_smoothing':
                setattr(args, r, 0.0)
            elif r == 'no_separator':
                setattr(args, r, True)  # old models don't use a separator
            else:
                # use default value
                setattr(args, r, None)

        # backward compatibility for models trained with genienlp before NED Refactoring (2)
        if args.max_features_size is None:
            if hasattr(args, 'ned_features_size'):
                setattr(args, 'max_features_size', args.ned_features_size)
            else:
                setattr(args, 'max_features_size', 0)
        if args.ned_domains is None:
            if hasattr(args, 'almond_domains'):
                setattr(args, 'ned_domains', args.almond_domains)
            else:
                setattr(args, 'ned_domains', [])
        if args.add_entities_to_text is None:
            if hasattr(args, 'add_types_to_text'):
                setattr(args, 'add_entities_to_text', args.add_types_to_text)
            else:
                setattr(args, 'add_entities_to_text', 'off')
        if args.entity_attributes is None:
            if hasattr(args, 'ned_features'):
                setattr(args, 'entity_attributes', args.ned_features)
            else:
                setattr(args, 'entity_attributes', [])
        if args.ned_normalize_types is None:
            if hasattr(args, 'bootleg_post_process_types') and args.bootleg_post_process_types:
                setattr(args, 'ned_normalize_types', 'soft')
        else:
            setattr(args, 'ned_normalize_types', 'off')

        args.dropout_ratio = 0.0
        args.verbose = False

    args.best_checkpoint = os.path.join(args.path, args.checkpoint_name)
