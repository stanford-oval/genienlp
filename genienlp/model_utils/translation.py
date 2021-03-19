import logging
import re
import numpy as np
import torch

from transformers import SPIECE_UNDERLINE

logger = logging.getLogger(__name__)

def compute_attention(sample_layer_attention, att_pooling, dim=0):
    # pool attention vectors across heads
    sample_layer_attention_pooled = None
    if att_pooling == 'mean':
        sample_layer_attention_pooled = torch.mean(sample_layer_attention, dim=dim, keepdim=False)
    elif att_pooling == 'max':
        sample_layer_attention_pooled = torch.max(sample_layer_attention, dim=dim, keepdim=False)[0]
    
    return sample_layer_attention_pooled

LOG_EVERY = 5000
log_counter = 0

def do_log(counter):
    return not counter % LOG_EVERY

def replace_quoted_params(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled):
    # find positions of quotation marks in src and tgt
    src2tgt_mapping = {}
    src2tgt_mapping_index = {}
    
    # ALL TODOS
    # first replace double quotes with single quote "" --> "
    # remove pad tokens before printing
    # add space around quotation mark if not present already: "NUMBER_1 " --> " NUMBER_1 "
    global log_counter

    # Note: quotation marks are exclusively used to wrap parameters so just check if they are present in the target sentence
    src_quotation_symbols = ['"']
    tgt_quotation_symbols = ['"', '«', '»', '“', '„']
    
    src_spans_ind = [index for index, token in enumerate(src_tokens) if
                     any([symbol in token for symbol in src_quotation_symbols])]
    tgt_spans_ind = [index for index, token in enumerate(tgt_tokens) if
                     any([symbol in token for symbol in tgt_quotation_symbols])]

    tokenizer._decode_use_source_tokenizer = True
    src_strings = tokenizer.convert_tokens_to_string(src_tokens)
    tokenizer._decode_use_source_tokenizer = False
    tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)
    
    if len(src_spans_ind) % 2 != 0:
        if do_log(log_counter):
            logging.error(f'Corrupted span in src string: [{src_strings}]')
        log_counter += 1
        return tgt_strings, False
    if len(tgt_spans_ind) % 2 != 0:
        if do_log(log_counter):
            logging.error(f'Corrupted span in tgt string: [{tgt_strings}] with src string: [{src_strings}]\n'
                          'outputting example without reverting the parameter')
        log_counter += 1
        return tgt_strings, False
    
    # arrange spans and exclude quotation mark indices
    src_spans = [(src_spans_ind[i] + 1, src_spans_ind[i + 1] - 1) for i in range(0, len(src_spans_ind), 2)]
    tgt_spans = [(tgt_spans_ind[i] + 1, tgt_spans_ind[i + 1] - 1) for i in range(0, len(tgt_spans_ind), 2)]
    
    if len(src_spans) != len(tgt_spans):
        if do_log(log_counter):
            logging.error(f'Numbers of spans in tgt and src strings do not match: [{tgt_strings}], [{src_strings}]\n'
                          'outputting example without reverting the parameter')
        log_counter += 1
        return tgt_strings, False
    
    tgt_span_success = set()
    for src_idx, (beg, end) in enumerate(src_spans):
        i = beg
        tgt_span_idx = None
        while i <= end:
            max_tgt_att_idx = torch.argmax(sample_layer_attention_pooled[:, i]).item()
            
            # find span in tgt that contains this index
            for tgt_idx, (s1, s2) in enumerate(tgt_spans):
                if s1 <= max_tgt_att_idx <= s2 and (s1, s2) not in tgt_span_success:
                    tgt_span_idx = tgt_idx
                    src2tgt_mapping[(beg, end)] = (s1, s2)
                    src2tgt_mapping_index[src_idx] = tgt_span_idx
                    tgt_span_success.add((s1, s2))
                    break
            if tgt_span_idx is not None:
                break
            else:
                # span could not be found; check the next wordpiece
                i += 1
        
        if tgt_span_idx is None:
            if do_log(log_counter):
                logger.error(f'Could not find a corresponding span in tgt for ({beg}, {end}) src span in src string: [{src_strings}]')
            log_counter += 1
            return tgt_strings, False
    
    src_quoted_pattern_maybe_space = re.compile(r'[{0}]\s?([^{0}]*?)\s?[{0}]'.format(''.join(src_quotation_symbols)))
    tgt_quoted_pattern_maybe_space = re.compile(r'[{0}]\s?([^{0}]*?)\s?[{0}]'.format(''.join(tgt_quotation_symbols)))
    
    src_matches = list(re.finditer(src_quoted_pattern_maybe_space, src_strings))
    tgt_matches = list(re.finditer(tgt_quoted_pattern_maybe_space, tgt_strings))
    
    tgt2src_mapping_index = {v: k for k, v in src2tgt_mapping_index.items()}
    
    # move through characters
    tokens = []
    curr = 0
    for pos, match in enumerate(tgt_matches):
        start, end = match.span()
        if start > curr:
            tokens.append(tgt_strings[curr:start])
        replace_match = src_matches[tgt2src_mapping_index[pos]]
        tokens.append(replace_match.group(0))
        curr = end
    if curr < len(tgt_strings):
        tokens.append(tgt_strings[curr:])
    
    text = ' '.join(tokens)
    
    return text, True


def force_replace_quoted_params(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled):
    # find positions of quotation marks in src
    src2tgt_mapping = {}
    global log_counter
    
    src_spans_ind = [index for index, token in enumerate(src_tokens) if '"' in token]
    if hasattr(tokenizer, 'is_piece_fn'):
        tgt_is_piece = [int(tokenizer.is_piece_fn(token)) for token in tgt_tokens]
    else:
        # assume spm tokenizer
        tgt_is_piece = [int(not token.startswith(SPIECE_UNDERLINE)) for token in tgt_tokens]
    tgt_piece2word_mapping = list(np.cumsum(tgt_is_piece) - 1)
    
    if len(src_spans_ind) % 2 != 0:
        if do_log(log_counter):
            tokenizer._decode_use_source_tokenizer = True
            logging.error(f'Corrupted span in src string: [{tokenizer.convert_tokens_to_string(src_tokens)}]')
            tokenizer._decode_use_source_tokenizer = False
        log_counter += 1
        # this almost never happens but if it does it is usually because quotation is missing from the end of src_tokens
        # we temporary fix this by adding '"' to the end of src_tokens
        src_tokens += tokenizer.tokenize('"')
        src_spans_ind = [index for index, token in enumerate(src_tokens) if '"' in token]
    
    tokenizer._decode_use_source_tokenizer = True
    src_strings = tokenizer.convert_tokens_to_string(src_tokens)
    tokenizer._decode_use_source_tokenizer = False
    tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)
    
    # arrange spans and exclude quotation mark indices
    src_spans = [(src_spans_ind[i] + 1, src_spans_ind[i + 1] - 1) for i in range(0, len(src_spans_ind), 2)]
    
    for src_idx, (beg, end) in enumerate(src_spans):
        s1 = torch.argmax(sample_layer_attention_pooled[:, beg]).item()
        s2 = torch.argmax(sample_layer_attention_pooled[:, end]).item()
        
        # clamp values to max tgt_tokens length
        s1 = min(s1, len(tgt_tokens) - 1)
        s2 = min(s2, len(tgt_tokens) - 1)
        
        src2tgt_mapping[(beg, end)] = (s1, s2)
    
    quoted_pattern_maybe_space = re.compile(r'\"\s?([^"]*?)\s?\"')
    
    src_matches = list(re.finditer(quoted_pattern_maybe_space, src_strings))
    
    # update src2tgt_mapping to map to word indices in response
    for key, value in src2tgt_mapping.items():
        s1, s2 = value
        try:
            src2tgt_mapping[key] = (
                max(0, tgt_piece2word_mapping[s1] - 1),
                min(tgt_piece2word_mapping[s2] + 1,
                len(tgt_tokens))
            )
        except:
            raise ValueError(f'Corrupted span in tgt string: [{tgt_strings}] with src string: [{src_strings}]\n'
                             'outputting example without reverting the parameter')
    
    # move through words
    tgt_strings_words = tgt_strings.split(' ')
    tokens = []
    curr = 0
    for i, (key, value) in enumerate(src2tgt_mapping.items()):
        start, end = value
        if start > curr:
            tokens.extend(tgt_strings_words[curr:start])
        replace_match = src_matches[i]
        tokens.append(replace_match.group(0))
        curr = end
    if curr < len(tgt_strings_words):
        tokens.extend(tgt_strings_words[curr:])
    
    text = ' '.join(tokens)
    
    return text