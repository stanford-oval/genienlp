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
import sys
import time
from json.decoder import JSONDecodeError

import numpy as np
import torch
import ujson
from transformers import MarianConfig, MBartConfig
from transformers.models.mbart50.tokenization_mbart50 import FAIRSEQ_LANGUAGE_CODES as MBART50_FAIRSEQ_LANGUAGE_CODES
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES as NLLB_FAIRSEQ_LANGUAGE_CODES

from .data_utils.almond_utils import token_type_regex
from .data_utils.example import NumericalizedExamples
from .data_utils.iterator import LengthSortedIterator
from .model_utils.transformers_utils import MARIAN_GROUP_MEMBERS
from .tasks.generic_dataset import all_tokens_fn, input_tokens_fn

logger = logging.getLogger(__name__)

ENTITY_MATCH_REGEX = re.compile('^([A-Z].*)_[0-9]+$')
QUOTED_MATCH_REGEX = re.compile(' " (.*?) " ')


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


def make_data_loader(
    dataset, numericalizer, batch_size, device=None, train=False, return_original_order=False, batching_algorithm='sample'
):
    args = numericalizer.args
    all_features = NumericalizedExamples.from_examples(dataset, numericalizer)

    context_lengths = [ex.context.length for ex in all_features]
    answer_lengths = [ex.answer.length for ex in all_features]

    topN = args.log_n_longest
    logger.info(
        f'context lengths (min, mean, max, top{topN}): {np.min(context_lengths)}, {int(np.mean(context_lengths))},'
        f' {np.max(context_lengths)}, {np.sort(context_lengths)[-topN:][::-1]}'
    )
    logger.info(
        f'answer lengths (min, mean, max, top{topN}): {np.min(answer_lengths)}, {int(np.mean(answer_lengths))},'
        f' {np.max(answer_lengths)}, {np.sort(answer_lengths)[-topN:][::-1]}'
    )

    if train:
        sort_key_fn = dataset.sort_key_fn
        batch_size_fn = dataset.batch_size_fn
    else:
        sort_key_fn = getattr(dataset, 'eval_sort_key_fn', dataset.sort_key_fn)
        batch_size_fn = getattr(dataset, 'eval_batch_size_fn', dataset.batch_size_fn)

    if batch_size_fn == input_tokens_fn:
        min_batch_length = np.min(context_lengths)
    elif batch_size_fn == all_tokens_fn:
        min_batch_length = np.min(context_lengths) + np.min(answer_lengths)
    else:
        min_batch_length = 1

    min_output_length = numericalizer.args.min_output_length
    max_output_length = numericalizer.args.max_output_length

    if min_batch_length > batch_size:
        raise ValueError(
            f'The minimum batch length in your dataset is {min_batch_length} but your batch size is {batch_size}.'
            f' Thus no examples will be processed. Consider increasing batch_size'
        )
    if np.min(answer_lengths) < min_output_length:
        raise ValueError(
            f'The minimum output length in your dataset is {np.min(answer_lengths)} but you have set --min_output_length to {min_output_length}.'
            f' Consider reducing that'
        )
    if np.max(answer_lengths) > max_output_length:
        raise ValueError(
            f'The maximum output length in your dataset is {np.max(answer_lengths)} but you have set --max_output_length to {max_output_length}.'
            f' Consider increasing that'
        )

    model_input_max_length = numericalizer._tokenizer.model_max_length

    all_features_filtered = []
    # remove examples longer than model input length

    for ex in all_features:
        # Uncomment for debugging to print the long examples
        # if dataset.batch_size_fn([ex]) > model_input_max_length:
        #     print(ex)
        if batch_size_fn([ex]) < model_input_max_length:
            all_features_filtered.append(ex)

    length_diff = len(all_features) - len(all_features_filtered)
    if length_diff != 0:
        if args.filter_long_inputs:
            logger.info(
                f'Removed {length_diff} examples that were longer than required model input_max_length of {model_input_max_length}'
            )
        else:
            raise ValueError(
                'Encountered an example that is longer than required model input_max_length. Consider using --filter_long_inputs to filter these examples.'
            )

    all_features = all_features_filtered

    sampler = LengthSortedIterator(
        all_features,
        batch_size=batch_size,
        sort=bool(sort_key_fn),
        shuffle_and_repeat=train,
        sort_key_fn=sort_key_fn,
        batch_size_fn=batch_size_fn,
        groups=dataset.groups,
        batching_algorithm=batching_algorithm,
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
    example_ids, predictions, raw_predictions, answers, contexts, confidence_features, src_lang, tgt_lang, is_entities=False
):
    new_example_ids, new_predictions, new_raw_predictions, new_answers, new_contexts, new_confidence_features = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    cur_raw_pred, cur_context, cur_answer = [], [], []
    if is_entities:
        cur_pred = {}
        split_token = '#'
    else:
        cur_pred = []
        split_token = '@'

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
        if split_token in ex_id:
            id_, split_id = ex_id.rsplit(split_token, 1)
            cur_id = id_
            while id_ == cur_id:
                if is_entities:
                    cur_pred[ctxt] = list(set(pred))
                else:
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
                    if split_token in ex_id:
                        id_, split_id = ex_id.rsplit(split_token, 1)
                    else:
                        id_ = ex_id
                else:
                    break

            new_example_ids.append(cur_id)
            if is_entities:
                new_predictions.append([json.dumps(cur_pred, ensure_ascii=False)])
                new_raw_predictions.append([json.dumps(cur_pred, ensure_ascii=False)])
            else:
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
            cur_raw_pred, cur_context, cur_answer = [], [], []
            if is_entities:
                cur_pred = {}
            else:
                cur_pred = []

        else:
            new_example_ids.append(ex_id)
            if is_entities:
                new_predictions.append([json.dumps({ctxt: list(set(pred))}, ensure_ascii=False)])
                new_raw_predictions.append([json.dumps({ctxt: list(set(pred))}, ensure_ascii=False)])
            else:
                new_predictions.append(pred)
                new_raw_predictions.append(raw_pred)
            new_contexts.append(ctxt)
            new_answers.append(ans)
            new_confidence_features.append(cf_feat)
            i += 1

    return new_example_ids, new_predictions, new_raw_predictions, new_answers, new_contexts, new_confidence_features


def get_model_lang(orig_lang, mapping):
    for lang in mapping:
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

    # adjust src and tgt languages for different models
    if isinstance(config, MBartConfig):
        src_lang = get_model_lang(src_lang, MBART50_FAIRSEQ_LANGUAGE_CODES)
        tgt_lang = get_model_lang(tgt_lang, MBART50_FAIRSEQ_LANGUAGE_CODES)
    # Nllb subclasses M2M100 config so we should check tokenizer_class directly
    elif getattr(config, 'tokenizer_class', '') == 'NllbTokenizer':
        src_lang = get_model_lang(src_lang, NLLB_FAIRSEQ_LANGUAGE_CODES)
        tgt_lang = get_model_lang(tgt_lang, NLLB_FAIRSEQ_LANGUAGE_CODES)

    return src_lang, tgt_lang


def have_multilingual(task_names):
    return any(['multilingual' in name for name in task_names])


def load_config_file_to_args(args):
    if not hasattr(args, 'is_hf_model'):
        # --is_hf_model might not exist if this function is called by anything other than predict.py
        setattr(args, 'is_hf_model', False)

    if args.is_hf_model:
        # no config file found, treat `args.path` as a model name on HuggingFace model hub
        args.pretrained_model = args.path
        args.override_question = ""  # because HF models are trained without a separate question
        config = vars(args).copy()
    else:
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
        'num_labels',
        'crossner_domains',
        'override_valid_metrics',
        'eval_src_languages',
        'eval_tgt_languages',
        'log_n_longest',
    ]

    # train and predict scripts have these arguments in common. We use the values from train only if they are not provided in predict.
    # NOTE: do not set default values for these arguments in predict cause the defaults will always override training arguments
    overwrite = [
        'model',
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
        'align_span_symbol',
        'max_output_length',
        'min_output_length',
        'reduce_metrics',
        'database_dir',
        'e2e_dialogue_valid_subtasks',
        'e2e_dialogue_valid_submetrics',
        'e2e_dialogue_valid_subweights',
    ]
    for o in overwrite:
        if o not in args or getattr(args, o) is None:
            retrieve.append(o)

    # these are true/ false arguments
    overwrite_actions = [
        'do_alignment',
        'align_preserve_input_quotation',
        'align_remove_output_quotation',
        'e2e_dialogue_evaluation',
        'filter_long_inputs',
    ]
    for o in overwrite_actions:
        # if argument is True in predict overwrite train; if False retrieve from train
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
        elif r == 'align_span_symbol':
            setattr(args, r, '"')
        elif r == 'log_n_longest':
            setattr(args, r, 3)
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
        elif r == 'min_output_length':
            setattr(args, r, 3)
        else:
            # use default value
            setattr(args, r, None)

    if args.e2e_dialogue_valid_subtasks is None:
        setattr(args, 'e2e_dialogue_valid_subtasks', ['dst', 'api', 'da', 'rg'])
    if args.e2e_dialogue_valid_submetrics is None:
        setattr(args, 'e2e_dialogue_valid_submetrics', ['dst_em', 'em', 'da_em', 'casedbleu'])
    if args.e2e_dialogue_valid_subweights is None:
        setattr(args, 'e2e_dialogue_valid_subweights', [1.0, 1.0, 1.0, 1.0])

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
