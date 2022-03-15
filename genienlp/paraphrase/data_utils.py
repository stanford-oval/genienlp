import logging
import random
import sys

import numpy as np
import torch

from ..data_utils.almond_utils import (
    detokenize_cjk_chars,
    device_pattern,
    input_heuristics,
    is_entity,
    quoted_pattern_maybe_space,
)
from ..data_utils.progbar import progress_bar
from ..util import get_number_of_lines

logger = logging.getLogger(__name__)


def group_together(file_paths, num_samples):
    """ """
    for i in range(1, len(num_samples)):
        num_samples[i] *= num_samples[i - 1]
    all_lines = []
    for file_path in file_paths:
        lines = []
        with open(file_path) as f:
            for line in f:
                lines.append(line.strip())
        all_lines.append(lines)

    all_groups = []
    for i, lines in enumerate(all_lines):
        for group_idx in range(0, len(lines) // num_samples[i]):
            g = lines[group_idx * num_samples[i] : (group_idx + 1) * num_samples[i]]
            if len(all_groups) <= group_idx:
                all_groups.append(g)
            else:
                all_groups[group_idx].extend(g)
    return all_groups


def mask_tokens(inputs, labels, tokenizer, mlm_probability, mlm_ignore_index):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = mlm_ignore_index  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def add_special_tokens(model, tokenizer, additional_special_tokens, pad_token=None):
    """Add special tokens to the tokenizer and the model if they have not already been added."""
    ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': additional_special_tokens}
    if pad_token is not None:
        ATTR_TO_SPECIAL_TOKEN['pad_token'] = pad_token
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    if num_added_tokens > 0:
        logger.info('Added %d special tokens', num_added_tokens)
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def find_index(input_sequence, tokens):
    for i in range(len(input_sequence)):
        found = True
        j = 0
        while j < len(tokens):
            if input_sequence[i + j] == tokens[j]:
                j += 1
            else:
                found = False
                break
        if found:
            return i
    return -1


def is_in_span(index, all_entity_spans):
    for span in all_entity_spans:
        if span[0] <= index < span[1]:
            return True
    return False


def token_masking(input_sequence, mlm_probability, mask_token, thingtalk):

    input_tokens = input_sequence.split(' ')

    all_entity_spans = []
    all_device_tokens = []
    if thingtalk:
        all_entities = quoted_pattern_maybe_space.findall(thingtalk)

        for entity in all_entities:
            entity_tokens = entity.split(' ')
            beg = find_index(input_tokens, entity_tokens)
            if beg != -1:
                all_entity_spans.append((beg, beg + len(entity_tokens)))

        for token in device_pattern.findall(thingtalk):
            all_device_tokens.extend(token.split('.'))

    # don't mask first and last tokens
    for i in range(1, len(input_tokens) - 1):
        curr_token = input_tokens[i]
        if is_entity(curr_token) or curr_token in all_device_tokens or is_in_span(i, all_entity_spans):
            mlm_probability /= 0.9
            continue
        if random.random() < mlm_probability:
            input_tokens[i] = mask_token
    return ' '.join(input_tokens)


def token_deletion(input_sequence, mlm_probability, mask_token, thingtalk):

    input_sequence_masked = token_masking(input_sequence, mlm_probability, mask_token, thingtalk)

    # go through all the tokens, and delete the ones that were masked
    input_tokens = list(filter(lambda x: x != mask_token, input_sequence_masked))

    return ' '.join(input_tokens)


def text_infilling(input_sequence, num_text_spans, max_tries, mask_token, thingtalk):

    input_tokens = input_sequence.split(' ')

    all_entity_spans = []
    all_device_tokens = []
    if thingtalk:
        all_entities = quoted_pattern_maybe_space.findall(thingtalk)

        for entity in all_entities:
            entity_tokens = entity.split(' ')
            beg = find_index(input_tokens, entity_tokens)
            if beg != -1:
                all_entity_spans.append((beg, beg + len(entity_tokens)))

        for token in device_pattern.findall(thingtalk):
            all_device_tokens.extend(token.split('.'))

    num_successful_spans = 0
    num_tries = 0
    while num_successful_spans < num_text_spans and num_tries < max_tries:
        num_tries += 1

        num_tokens_to_mask = np.random.poisson(lam=3)
        mask_start_index = random.randint(0, len(input_tokens) - 1)

        if mask_start_index + num_tokens_to_mask > len(input_tokens):
            continue
        if input_tokens[mask_start_index] == mask_token:
            continue

        # don't mask first and last tokens
        if mask_start_index == 0 or mask_start_index == len(input_tokens) - 1:
            continue

        contains_crucial_token = False
        # check this span for a crucial token
        for j in range(0, num_tokens_to_mask):
            curr_token = input_tokens[mask_start_index + j]
            if is_entity(curr_token) or curr_token in all_device_tokens or is_in_span(mask_start_index + j, all_entity_spans):
                contains_crucial_token = True
                break
        if not contains_crucial_token:
            num_successful_spans += 1
            if num_tokens_to_mask + mask_start_index != len(input_tokens):
                input_tokens = (
                    input_tokens[:mask_start_index] + [mask_token] + input_tokens[mask_start_index + num_tokens_to_mask :]
                )
            else:
                input_tokens = input_tokens[:mask_start_index] + [mask_token]

    return ' '.join(input_tokens)


def sentence_permutation(input_sequence):
    input_tokens = input_sequence.split('.')
    random.shuffle(input_tokens)
    return ' '.join(input_tokens)


def document_rotation(input_sequence):
    input_tokens = input_sequence.split(' ')
    token_index_to_rotate = random.randint(0, len(input_tokens) - 1)
    input_tokens = input_tokens[token_index_to_rotate:] + input_tokens[:token_index_to_rotate]
    return ' '.join(input_tokens)


def create_features_from_tsv_file(
    file_path,
    tokenizer,
    input_column,
    gold_column,
    id_column,
    prompt_column,
    thingtalk_column,
    copy,
    sep_token_id,
    skip_heuristics,
    is_cased,
    model_type,
    src_lang,
    subsample,
    shuffle_input,
    task,
    model_input_prefix,
    max_input_length,
    mask_tokens,
    mask_token_prob,
    masking_token,
    infill_max_tries,
    delete_tokens,
    delete_token_prob,
    infill_text,
    num_text_spans,
    permute_sentences,
    rotate_sentence,
):
    """
    Read a tsv file (this includes a text file with one example per line) and returns input features that the model needs
    Outputs:

    """
    all_input_sequences = []
    all_input_sequence_lengths = []
    all_example_ids = []
    all_context_ids = []
    estimated_output_lengths = []
    all_golds = []
    reverse_maps = []
    all_prompt_ids = []

    if file_path is not None:
        number_of_lines = get_number_of_lines(file_path)
        disable_progbar = False
        input_file = open(file_path)
    else:
        number_of_lines = 1
        disable_progbar = True
        input_file = sys.stdin

    line_count = 0

    if shuffle_input:
        all_lines = []
        for line in input_file:
            all_lines.append(line)
        random.shuffle(all_lines)
        all_lines = iter(all_lines)
    else:
        all_lines = input_file

    for line in progress_bar(all_lines, desc='Reading Input File', total=number_of_lines, disable=disable_progbar):
        row = [r.strip() for r in line.split('\t')]
        input_sequence = row[input_column]
        gold = row[gold_column]
        if id_column is not None:
            id_ = row[id_column]
        else:
            id_ = line_count
        all_example_ids.append(id_)
        if not skip_heuristics:
            gold, _ = input_heuristics(gold, None, is_cased, keep_special_tokens=True, keep_tokenized=True)
        all_golds.append(gold)
        thingtalk = row[thingtalk_column] if thingtalk_column is not None else None
        if skip_heuristics:
            reverse_maps.append({})
        else:
            input_sequence, reverse_map = input_heuristics(input_sequence, thingtalk, is_cased)
            reverse_maps.append(reverse_map)

        if mask_tokens:
            input_sequence = token_masking(input_sequence, mask_token_prob, masking_token, thingtalk)
        if delete_tokens:
            input_sequence = token_deletion(input_sequence, delete_token_prob, masking_token, thingtalk)
        if infill_text:
            input_sequence = text_infilling(input_sequence, num_text_spans, infill_max_tries, masking_token, thingtalk)
        if permute_sentences:
            input_sequence = sentence_permutation(input_sequence)
        if rotate_sentence:
            input_sequence = document_rotation(input_sequence)

        # add model specific prefix
        input_sequence = model_input_prefix + input_sequence

        if model_type in ['mbart', 'mbart50']:
            # just make sure source language is used when tokenizing input sentence
            # tokenizer takes care of adding language code at the end of the sentence
            tokenizer.src_lang = src_lang
            tokenizer.cur_lang_code = tokenizer.lang_code_to_id[src_lang]

        input_sequence = detokenize_cjk_chars(input_sequence)
        input_sequence_ids = tokenizer.encode(input_sequence, add_special_tokens=True)

        if len(input_sequence_ids) > max_input_length:
            # keep eos token. This is approximate cause there might be other special tokens appended,
            # but in practice we rarely face examples longer than 512 sub tokens
            input_sequence_ids = input_sequence_ids[: max_input_length - 2] + input_sequence_ids[-1:]

        prompt_ids = []  # includes the first few tokens of the output
        if prompt_column is not None and len(row) > prompt_column:
            prompt = row[prompt_column]
            if not skip_heuristics:
                prompt, _ = input_heuristics(prompt, thingtalk, is_cased)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if copy > 0:
            assert len(prompt_ids) == 0
            prompt_ids = input_sequence_ids[0 : min(copy, len(input_sequence_ids) - 1)]
        all_prompt_ids.append(prompt_ids)

        # TODO problemtaic for marian and bart models
        if task != 'translate':
            context_ids = input_sequence_ids + [sep_token_id] + prompt_ids
        else:
            context_ids = input_sequence_ids

        all_input_sequences.append(input_sequence)
        all_input_sequence_lengths.append(len(input_sequence_ids))
        all_context_ids.append(context_ids)
        estimated_output_lengths.append(len(input_sequence_ids) - len(prompt_ids))

        line_count += 1
        if line_count >= subsample:
            break
    logger.info("Input has {} examples; and we subsampled {} examples".format(number_of_lines, line_count))

    if file_path is not None:
        input_file.close()

    return (
        all_input_sequences,
        all_input_sequence_lengths,
        all_example_ids,
        all_context_ids,
        estimated_output_lengths,
        all_golds,
        reverse_maps,
        all_prompt_ids,
    )
