import sys
import re
import random

import torch
import logging
import numpy as np

from ..data_utils.progbar import progress_bar
from ..util import detokenize, tokenize, lower_case, SpecialTokenMap, remove_thingtalk_quotes

from genienlp.util import get_number_of_lines
from ..tasks.almond.utils import is_entity, quoted_pattern_maybe_space, device_pattern


logger = logging.getLogger(__name__)


special_pattern_mapping = [
    SpecialTokenMap('PHONE_NUMBER_([0-9]+)', ['888-8888', '777-8888']),
    SpecialTokenMap('NUMBER_([0-9]+)', ['2', '3'], [['2', 'two'], ['3', 'three']]),
    SpecialTokenMap('PATH_NAME_([0-9]+)', ['my1folder', 'my2folder']),
    SpecialTokenMap('TIME_([0-9]+)', ['5p.m.', '2p.m.'], [['5 pm', '5pm', '5:00 pm', '5:00pm', '5p.m.', '5 p.m.', '5:00 p.m.', '5:00', 'five o\'clock', 'five'],
                                                            ['2 pm', '2pm', '2:00 pm', '2:00pm', '2p.m.', '2 p.m.', '2:00 p.m.', '2:00', 'two o\'clock', 'two']]),
    SpecialTokenMap('EMAIL_ADDRESS_([0-9]+)', ['e1@example.com', 'e2@example.com']),
    SpecialTokenMap('URL_([0-9]+)', ['my1site.com', 'my2site.com']),
    SpecialTokenMap('DATE_([0-9]+)', ['5-6-2015', '8-3-2016']),
    SpecialTokenMap('CURRENCY_([0-9]+)', ['$12', '$13'], [['$12', 'twelve dollars', '12 dollars', '$ 12', '$ 12.00', '12.00', '12'],
                                                          ['$13', 'thirteen dollars', '13 dollars', '$ 13', '$ 13.00', '13.00', '13']]),
    SpecialTokenMap('DURATION_([0-9]+)', ['5 weeks', '6 weeks'], [['5 weeks', 'five weeks'], ['6 weeks', 'six weeks']]),
    SpecialTokenMap('LOCATION_([0-9]+)', ['locatio1n', 'locatio2n'], [['locatio1n', 'locat1n'], ['locatio2n', 'locat2n']]),
    SpecialTokenMap('QUOTED_STRING_([0-9]+)', lambda x: 'Chinese', lambda x: ['Chinese', 'chinese', 'china']), # TODO change to be more general than cuisine
    SpecialTokenMap('GENERIC_ENTITY_uk.ac.cam.multiwoz.Restaurant:Restaurant_([0-9]+)', ["restaurant1", "restaurant2", "restaurant3"]) # TODO the only reason we can get away with this unnatural replacement is that actual backward is not going to be called for this
]


def group_together(file_paths, num_samples):
    """
    """
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
            g = lines[group_idx * num_samples[i]:(group_idx + 1) * num_samples[i]]
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
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': additional_special_tokens}
    if pad_token is not None:
        ATTR_TO_SPECIAL_TOKEN['pad_token'] = pad_token
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        logger.info('Added %d special tokens', num_added_tokens)
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def has_match(sentence, all_entities):
    for entity in all_entities:
        j = 0
        entity_tokenized = entity.split(' ')
        found = True
        while j < len(entity_tokenized):
            if sentence[j] == entity_tokenized[j]:
                j += 1
            else:
                found = False
                break
        if found:
            return True
    return False
    
def token_masking(input_sequence, mlm_probability, mask_token, thingtalk):
    
    all_entities = []
    all_device_tokens = []
    if thingtalk:
        all_entities = quoted_pattern_maybe_space.findall(thingtalk)
        for token in device_pattern.findall(thingtalk):
            all_device_tokens.extend(token.split('.'))
    
    input_tokens = input_sequence.split(' ')
    input_length = len(input_tokens)
    
    # don't mask first and last tokens
    for i in range(1, input_length-1):
        if is_entity(input_tokens[i]) or input_tokens[i] in all_device_tokens or has_match(input_tokens[i:], all_entities):
            mlm_probability /= 0.9
            continue
        if random.random() < mlm_probability:
            input_tokens[i] = mask_token
    return ' '.join(input_tokens)

def token_deletion(input_sequence, mlm_probability, mask_token, thingtalk):
    all_entities = []
    all_device_tokens = []
    if thingtalk:
        all_entities = quoted_pattern_maybe_space.findall(thingtalk)
        for token in device_pattern.findall(thingtalk):
            all_device_tokens.extend(token.split('.'))
    
    input_tokens = input_sequence.split(' ')
    input_length = len(input_tokens)
    
    # don't mask first and last tokens
    for i in range(1, input_length-1):
        if is_entity(input_tokens[i]) or input_tokens[i] in all_device_tokens or has_match(input_tokens[i:], all_entities):
            mlm_probability /= 0.9
            continue
        if random.random() < mlm_probability:
            # temporarily put a mask token here
            input_tokens[i] = mask_token
    
    # go through all the tokens, and delete the ones that were masked
    input_tokens = list(filter(lambda x: x != mask_token, input_tokens))

    return ' '.join(input_tokens)

def text_infilling(input_sequence, num_text_spans, mask_token, thingtalk):
    all_entities = []
    all_device_tokens = []
    if thingtalk:
        all_entities = quoted_pattern_maybe_space.findall(thingtalk)
        for token in device_pattern.findall(thingtalk):
            all_device_tokens.extend(token.split('.'))

    input_tokens = input_sequence.split(' ')

    num_successful_spans = 0
    while num_successful_spans < num_text_spans:
        num_tokens_to_mask = np.random.poisson(lam=3)
        mask_start_index = random.randint(0, len(input_tokens) - 1)

        if mask_start_index + num_tokens_to_mask > len(input_tokens):
            continue
        if input_tokens[mask_start_index] == mask_token:
            continue
        
        contains_crucial_token = False
        # check this span for a crucial token
        for j in range(0, num_tokens_to_mask):
            curr_token = input_tokens[mask_start_index + j]
            if is_entity(curr_token) or curr_token in all_device_tokens or has_match(input_tokens[j + mask_start_index:], all_entities):
                contains_crucial_token = True
                break
        if not contains_crucial_token:
            num_successful_spans += 1
            if num_tokens_to_mask + mask_start_index != len(input_tokens):
                input_tokens = input_tokens[:mask_start_index] + [mask_token] + input_tokens[mask_start_index + num_tokens_to_mask:]
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

def create_features_from_tsv_file(file_path, tokenizer, input_column, gold_column, id_column, prompt_column, thingtalk_column,
                                  copy, sep_token_id, skip_heuristics, is_cased, model_type,
                                  src_lang, subsample, task, model_input_prefix,
                                  mask_tokens, mask_token_prob, masking_token,
                                  delete_tokens, delete_token_prob,
                                  infill_text, num_text_spans,
                                  permute_sentences,
                                  rotate_sentence):
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
    for line in progress_bar(input_file, desc='Reading Input File', total=number_of_lines, disable=disable_progbar):
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
            input_sequence = text_infilling(input_sequence, num_text_spans, masking_token, thingtalk)
        if permute_sentences:
            input_sequence = sentence_permutation(input_sequence)
        if rotate_sentence:
            input_sequence = document_rotation(input_sequence)
        
        # add model specific prefix
        input_sequence = model_input_prefix + input_sequence
        
        if model_type == 'mbart':
            # just make sure source language is used when tokenizing input sentence
            # tokenizer takes care of adding language code at the end of the sentence
            tokenizer.cur_lang_code = tokenizer.lang_code_to_id[src_lang]
            
        input_sequence_ids = tokenizer.encode(input_sequence, add_special_tokens=True)
        
        prompt_ids = [] # includes the first few tokens of the output
        if prompt_column is not None and len(row) > prompt_column:
            prompt = row[prompt_column]
            if not skip_heuristics:
                prompt, _ = input_heuristics(prompt, thingtalk, is_cased)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if copy > 0:
            assert len(prompt_ids) == 0
            prompt_ids = input_sequence_ids[0 : min(copy, len(input_sequence_ids)-1)]
        all_prompt_ids.append(prompt_ids)
        
        #TODO problemtaic for marian and bart models
        if task != 'translate':
            context_ids = input_sequence_ids + [sep_token_id] + prompt_ids
        else:
            context_ids = input_sequence_ids
        
        all_input_sequences.append(input_sequence)
        all_input_sequence_lengths.append(len(input_sequence_ids))
        all_context_ids.append(context_ids)
        estimated_output_lengths.append(len(input_sequence_ids)-len(prompt_ids))
        
        line_count += 1
        if line_count >= subsample:
            break
    logger.info("Input has {} examples; and we subsampled {} examples".format(number_of_lines, line_count))
    
    if file_path is not None:
        input_file.close()

    return all_input_sequences, all_input_sequence_lengths, all_example_ids, all_context_ids, estimated_output_lengths, all_golds, reverse_maps, all_prompt_ids


def is_question(sentence: str):
    question_words = ['which', 'what', 'where', 'how', 'who', 'when', 'is', 'are', 'am',
                      'can', 'could', 'would', 'will', 'have', 'did', 'do', 'does', 'no is', 'yes is']
    for w in question_words:
        if sentence.startswith(w+' '):
            return True
    return False

def input_heuristics(s: str, thingtalk=None, is_cased=False, keep_special_tokens=False, keep_tokenized=False):
    """
    Changes the input string so that it is closer to what the pre-trained language models have seen during their training.
    Outputs:
        s: the new string
        reverse_map: a list of special tokens. Can be used to recover the original special_tokens in the string
    """
    s = s.strip()
    s = tokenize(s)

    # Put question mark at the end whenever necessary.
    sentences = [sentence.strip() for sentence in re.split('\s+([.?!:])\s*', s) if len(sentence) > 0]
    # logger.info('sentences = %s', sentences)
    for idx in range(len(sentences)):
        if sentences[idx] in ['.', '?' , '!', ':']:
            continue
        if idx == len(sentences)-1 or sentences[idx+1] not in ['.', '?', '!', ':']:
            # add the missing punctuation
            if is_question(sentences[idx]):
                sentences[idx] = sentences[idx] + '?'
            else:
                sentences[idx] = sentences[idx] + '.'
        else:
            if is_question(sentences[idx]):
                assert sentences[idx+1] in ['.', '?', '!', ':']
                sentences[idx+1] = '?'

        if is_cased:
            # capitalize the first word and parameters
            if thingtalk:
                _, parameters = remove_thingtalk_quotes(thingtalk)
                # logger.info('parameters = ', parameters)
                for p in parameters:
                    capitalized_p = ' '.join([t[0].upper()+t[1:] for t in p.split()])
                    sentences[idx] = sentences[idx].replace(p, capitalized_p)
            sentences[idx] = sentences[idx].replace(' i ', ' I ')
            sentences[idx] = sentences[idx][0].upper()+sentences[idx][1:]
            
    s = ' '.join(sentences)
    if not keep_tokenized:
        s = detokenize(s)
    
    if not is_cased:
        s = lower_case(s)

    # replace special tokens with natural-looking examples
    reverse_map = []
    if not keep_special_tokens:
        for spm in special_pattern_mapping:
            s, r = spm.forward(s)
            reverse_map.extend(r)

    return s, reverse_map

def output_heuristics(s: str, reverse_map: list):
    for spm, occurance in reverse_map:
        s = spm.backward(s, occurance)

    s = tokenize(s)
    s = lower_case(s)
    return s
