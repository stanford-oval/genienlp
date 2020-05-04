import os
import glob
import sys
import re

from tqdm import tqdm
import torch
import logging

from ..util import detokenize, tokenize, lower_case, SpecialTokenMap, remove_thingtalk_quotes

from genienlp.paraphrase.dataset import TextDataset
from genienlp.util import get_number_of_lines

logger = logging.getLogger(__name__)


special_pattern_mapping = [
    SpecialTokenMap('PHONE_NUMBER_([0-9]+)', ['888-8888', '777-8888']),
    SpecialTokenMap('NUMBER_([0-9]+)', ['2', '3'], [['2', 'two'], ['3', 'three']]),
    SpecialTokenMap('PATH_NAME_([0-9]+)', ['my1folder', 'my2folder']),
    SpecialTokenMap('TIME_([0-9]+)', ['1p.m.', '2p.m.'], [['1 pm', '1pm', '1:00 pm', '1:00pm', '1p.m.', '1 p.m.', '1:00 p.m.', '1:00', 'one o\'clock', 'one'],
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



def load_and_cache_examples(args, tokenizer, evaluate=False, aux=False):
    if evaluate:
        if aux:
            file_path = args.aux_eval_data_file
        else:
            file_path = args.eval_data_file
    else:
        file_path = args.train_data_file
    dataset = TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size, evaluate=evaluate)
    return dataset


def mask_tokens(inputs, labels, tokenizer, args):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def remove_thingtalk_quotes(thingtalk):
    while True:
        l1 = thingtalk.find('"')
        if l1 < 0:
            break
        l2 = thingtalk.find('"', l1+1)
        assert l2 >= 0
        thingtalk = thingtalk[:l1] + '<temp>' + thingtalk[l2+1:]
    return thingtalk

def get_inbetween_tokens(train_data_file, start_token, end_token):
    new_tokens = set()
    with open(train_data_file, encoding="utf-8") as f:
        for line in tqdm(f, desc='Tokenizing'):
            line = remove_thingtalk_quotes(line)
            line = line[line.find(start_token)+len(start_token):line.find(end_token)]
            new_tokens.update(line.split())
    return new_tokens

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


def create_features_from_tsv_file(file_path, tokenizer, input_column, gold_column, prompt_column, copy, thingtalk_column, sep_token_id,
                                  skip_heuristics, is_cased, model_type):
    """
    Read a tsv file (this includes a text file with one example per line) and returns input features that the model needs
    Outputs:

    """
    all_input_sequences = []
    all_input_sequence_lengths = []
    all_context_tokens = []
    estimated_output_lengths = []
    all_golds = []
    reverse_maps = []
    all_prompt_tokens = []

    if file_path is not None:
        number_of_lines = get_number_of_lines(file_path)
        disable_tqdm = False
        input_file = open(file_path)
    else:
        number_of_lines = 1
        disable_tqdm = True
        input_file = sys.stdin


    for line in tqdm(input_file, desc='Reading Input File', total=number_of_lines, disable=disable_tqdm):
        row = [r.strip() for r in line.split('\t')]
        input_sequence = row[input_column]
        gold = row[gold_column]
        # logger.info('gold (before heuristics) = %s', gold)
        if not skip_heuristics:
            gold, _ = input_heuristics(gold, None, is_cased, keep_special_tokens=True, keep_tokenized=True)
        # logger.info('gold (after heuristics) = %s', gold)
        all_golds.append(gold)
        if skip_heuristics:
            reverse_maps.append({})
        else:
            thingtalk = row[thingtalk_column] if thingtalk_column is not None else None
            # logger.info('input_sequence (before heuristics) = %s', input_sequence)
            input_sequence, reverse_map = input_heuristics(input_sequence, thingtalk, is_cased)
            # logger.info('input_sequence (after heuristics) = %s', input_sequence)
            reverse_maps.append(reverse_map)
        input_sequence_tokens = tokenizer.encode(input_sequence, add_special_tokens=True)
        
        prompt_tokens = [] # includes the first few tokens of the output
        if prompt_column is not None and len(row) > prompt_column:
            prompt = row[prompt_column]
            if not skip_heuristics:
                prompt, _ = input_heuristics(prompt, thingtalk, is_cased)
                # logger.info('prompt = %s', prompt)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if copy > 0:
            assert len(prompt_tokens) == 0
            prompt_tokens = input_sequence_tokens[0 : min(copy, len(input_sequence_tokens)-1)]
        all_prompt_tokens.append(prompt_tokens)
        context_tokens = input_sequence_tokens + [sep_token_id] + prompt_tokens
        all_input_sequences.append(input_sequence)
        all_input_sequence_lengths.append(len(input_sequence_tokens))
        all_context_tokens.append(context_tokens)
        estimated_output_lengths.append(len(input_sequence_tokens)-len(prompt_tokens))
    
    if file_path is not None:
        input_file.close()

    return all_input_sequences, all_input_sequence_lengths, all_context_tokens, estimated_output_lengths, all_golds, reverse_maps, all_prompt_tokens


def is_question(sentence: str):
    question_words = ['which', 'what', 'where', 'how', 'who', 'when', 'is', 'are', 'am', \
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
    reverse_map = []
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

    # replace special tokens with natural-looking exmaples
    reverse_map = []
    if not keep_special_tokens:
        for spm in special_pattern_mapping:
            s, r = spm.forwad(s)
            reverse_map.extend(r)

    return s, reverse_map

def output_heuristics(s: str, reverse_map: list):
    for spm, occurance in reverse_map:
        s = spm.backward(s, occurance)

    s = tokenize(s)
    s = lower_case(s)
    return s
