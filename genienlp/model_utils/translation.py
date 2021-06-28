# Copyright 2021 The Board of Trustees of the Leland Stanford Junior University
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
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
    global log_counter

    # Note: quotation marks are exclusively used to wrap parameters so just check if they are present in the target sentence
    src_quotation_symbols = ['"']
    tgt_quotation_symbols = ['"', '«', '»', '“', '„']

    tokenizer._decode_use_source_tokenizer = False

    tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)
    for symbol in tgt_quotation_symbols:
        # 1) replace multiple quotes with single quote
        tgt_strings = re.sub(rf"{symbol}+", rf"{symbol}", tgt_strings)

        # 2) add space around every symbol
        tgt_strings = re.sub(rf"{symbol}", rf" {symbol} ", tgt_strings)

        # 3) remove any double spaces
        tgt_strings = re.sub(r"\s\s", " ", tgt_strings)

    with tokenizer.as_target_tokenizer():
        tgt_tokens = tokenizer.tokenize(tgt_strings)

    src_spans_ind = [
        index for index, token in enumerate(src_tokens) if any([symbol in token for symbol in src_quotation_symbols])
    ]
    tgt_spans_ind = [
        index for index, token in enumerate(tgt_tokens) if any([symbol in token for symbol in tgt_quotation_symbols])
    ]

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
            logging.error(f'Corrupted span in tgt string: [{tgt_strings}] with src string: [{src_strings}]\n')
        log_counter += 1
        return tgt_strings, False

    # arrange spans and exclude quotation mark indices
    src_spans = [(src_spans_ind[i] + 1, src_spans_ind[i + 1] - 1) for i in range(0, len(src_spans_ind), 2)]
    tgt_spans = [(tgt_spans_ind[i] + 1, tgt_spans_ind[i + 1] - 1) for i in range(0, len(tgt_spans_ind), 2)]

    if len(src_spans) != len(tgt_spans):
        if do_log(log_counter):
            logging.error(f'Numbers of spans in tgt and src strings do not match: [{tgt_strings}], [{src_strings}]\n')
        log_counter += 1
        return tgt_strings, False

    tgt_span_success = set()
    for src_idx, (beg, end) in enumerate(src_spans):
        i = beg
        tgt_span_idx = None
        while i <= end:
            max_tgt_att_idx = torch.argmax(sample_layer_attention_pooled[:, i]).item()

            # find span in tgt that contains this index; -1 and +1 to include target quotations marks
            for tgt_idx, (s1, s2) in enumerate(tgt_spans):
                if s1 - 1 <= max_tgt_att_idx <= s2 + 1 and (s1, s2) not in tgt_span_success:
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
                logger.error(
                    f'Could not find a corresponding span in tgt for ({beg}, {end}) src span in src string: [{src_strings}]'
                )
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

    src_quotation_symbols = ['"']
    tgt_quotation_symbols = ['"', '«', '»', '“', '„']

    global log_counter

    # replace double quotes with single quote
    for symbol in tgt_quotation_symbols:
        tgt_tokens = ' '.join(tgt_tokens).replace(f'{symbol}{symbol}', f'{symbol}').split(' ')

    src_spans_ind = [
        index for index, token in enumerate(src_tokens) if any([symbol in token for symbol in src_quotation_symbols])
    ]
    tgt_is_not_piece = [int(not tokenizer.is_piece_fn(token)) for token in tgt_tokens]
    tgt_piece2word_mapping = list(np.cumsum(tgt_is_not_piece) - 1)

    if len(src_spans_ind) % 2 != 0:
        if do_log(log_counter):
            tokenizer._decode_use_source_tokenizer = True
            raise ValueError(f'Corrupted span in src string: [{tokenizer.convert_tokens_to_string(src_tokens)}]')

    tokenizer._decode_use_source_tokenizer = True
    src_strings = tokenizer.convert_tokens_to_string(src_tokens)
    tokenizer._decode_use_source_tokenizer = False
    tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)

    # arrange spans but DO NOT exclude quotation mark indices
    src_spans = [(src_spans_ind[i], src_spans_ind[i + 1]) for i in range(0, len(src_spans_ind), 2)]

    for src_idx, (beg, end) in enumerate(src_spans):
        s1 = torch.argmax(sample_layer_attention_pooled[:, beg]).item()
        s2 = torch.argmax(sample_layer_attention_pooled[:, end]).item()

        # clamp values to max tgt_tokens length
        s1 = min(s1, len(tgt_tokens) - 1)
        s2 = min(s2, len(tgt_tokens) - 1)

        # switch tgt begin and end indices
        if s1 > s2:
            s1, s2 = s2, s1

        src2tgt_mapping[(beg, end)] = (s1, s2)

    quoted_pattern_maybe_space = re.compile(r'\"\s?([^"]*?)\s?\"')

    src_matches = list(re.finditer(quoted_pattern_maybe_space, src_strings))

    # update src2tgt_mapping to map to word indices in response
    for key, value in src2tgt_mapping.items():
        s1, s2 = value
        src2tgt_mapping[key] = (
            max(0, tgt_piece2word_mapping[s1]),
            min(tgt_piece2word_mapping[s2], len(tgt_tokens)),
        )

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
        # +1 since it's inclusive
        curr = end + 1
    if curr < len(tgt_strings_words):
        tokens.extend(tgt_strings_words[curr:])

    text = ' '.join(tokens)

    return text
