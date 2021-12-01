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
from collections import OrderedDict

import numpy as np
import torch
from dateparser.conf import Settings
from num2words import CONVERTER_CLASSES, num2words
from transformers import SPIECE_UNDERLINE, M2M100Tokenizer

from genienlp.data_utils.almond_utils import NUMBER_MAPPING

logger = logging.getLogger(__name__)


def find_overlap(start, end, used_spans):
    for i, span in enumerate(used_spans):
        span_start, span_end = span[0], span[1]
        if start <= span_end and end >= span_start:
            return i
    return -1


def count_substring(words, substring_words):
    count = 0
    beg_indices = []
    for i in range(len(words)):
        if words[i].lower() == substring_words[0].lower():
            k = 0
            while k < len(substring_words) and i + k < len(words):
                if words[i + k].lower() == substring_words[k].lower():
                    k += 1
                else:
                    break
            if k == len(substring_words):
                count += 1
                beg_indices.append(i)
    return count, beg_indices


def compute_attention(sample_layer_attention, att_pooling, dim=0):
    # pool attention vectors across heads
    sample_layer_attention_pooled = None
    if att_pooling == 'mean':
        sample_layer_attention_pooled = torch.mean(sample_layer_attention, dim=dim, keepdim=False)
    elif att_pooling == 'max':
        sample_layer_attention_pooled = torch.max(sample_layer_attention, dim=dim, keepdim=False)[0]

    return sample_layer_attention_pooled


def return_token_word_mapping(tokens, tokenizer):
    is_not_piece = [int(not tokenizer.is_piece_fn(token)) for token in tokens]
    token2word_mapping = list(np.cumsum(is_not_piece) - 1)
    word2token_span_mapping = OrderedDict()
    for i, j in enumerate(token2word_mapping):
        if j not in word2token_span_mapping:
            word2token_span_mapping[j] = [i, i]
        else:
            word2token_span_mapping[j][1] = i

    return token2word_mapping, word2token_span_mapping


def align_and_replace(
    src_tokens,
    tgt_tokens,
    sample_layer_attention_pooled,
    src_spans,
    tgt_lang,
    tokenizer,
    remove_output_quotation,
    date_parser=None,
):
    src_quotation_symbol = '"'

    # M2M100Tokenizer has missing tokens in its fixed vocabulary and encodes them as unknown (https://github.com/pytorch/fairseq/issues/3463)
    # until that's fixed we treat unknown tokens as individual words by prepending SPIECE_UNDERLINE
    if isinstance(tokenizer, M2M100Tokenizer):
        src_tokens = [token if token != tokenizer.unk_token else SPIECE_UNDERLINE + token for token in src_tokens]

    tokenizer._decode_use_source_tokenizer = True
    src_string = tokenizer.convert_tokens_to_string(src_tokens)
    tokenizer._decode_use_source_tokenizer = False
    tgt_string = tokenizer.convert_tokens_to_string(tgt_tokens)

    src_words = src_string.split(' ')
    tgt_words = tgt_string.split(' ')

    if len(src_spans) % 2 != 0:
        raise ValueError(f'Corrupted span in src string: [{src_string}]')
    src_spans = [(src_spans[i], src_spans[i + 1]) for i in range(0, len(src_spans), 2)]
    src_matches = [tuple(src_words[beg : end + 1]) for beg, end in src_spans]
    src_matches_counter = OrderedDict()
    for match, spans in zip(src_matches, src_spans):
        src_matches_counter.setdefault(match, []).append(spans)

    src_token2word_mapping, src_word2token_span_mapping = return_token_word_mapping(src_tokens, tokenizer)
    tgt_token2word_mapping, tgt_word2token_span_mapping = return_token_word_mapping(tgt_tokens, tokenizer)

    src_token_spans = [(src_word2token_span_mapping[beg][0], src_word2token_span_mapping[end][1]) for beg, end in src_spans]
    src2tgt_token_mapping = {}

    # if translation preserved input entities we won't align them anymore
    for cur_match, spans in src_matches_counter.items():
        # expanded_matches keep current match and any possible known transformation of current match
        expanded_matches = [cur_match]

        # translation turned digit into words
        if len(cur_match) == 1 and cur_match[0].isdigit():
            # int converts arabic digits to english
            match = int(cur_match[0])
            if tgt_lang in CONVERTER_CLASSES or tgt_lang[:2] in CONVERTER_CLASSES:
                expanded_matches.append([num2words(match, lang=tgt_lang, to='cardinal')])

            if any(tgt_lang.startswith(lang) for lang in ['fa', 'ar']):
                match = str(match)
                src_numbers = NUMBER_MAPPING['en']
                tgt_numbers = NUMBER_MAPPING['fa']
                if match in src_numbers:
                    index = src_numbers.index(match)
                    tgt_number = tgt_numbers[index]
                    expanded_matches.append([tgt_number])

        # find translation of dates
        elif date_parser:
            expanded_matches.append(date_parser.translate(' '.join(cur_match), settings=Settings()).split(' '))

        for match in expanded_matches:
            count, beg_indices = count_substring(tgt_words, match)
            # we found matching spans in target so just update tgt_token_spans to make sure we don't overwrite them later
            if count == len(spans):
                for span, id_ in zip(spans, beg_indices):
                    beg_word, end_word = id_, id_ + len(match) - 1
                    beg_tgt_token, end_tgt_token = (
                        tgt_word2token_span_mapping[beg_word][0],
                        tgt_word2token_span_mapping[end_word][1],
                    )
                    beg_src_token, end_sc_token = (
                        src_word2token_span_mapping[span[0]][0],
                        src_word2token_span_mapping[span[1]][1],
                    )
                    src2tgt_token_mapping[(beg_src_token, end_sc_token)] = (beg_tgt_token, end_tgt_token)
                    break

    for src_beg, src_end in src_token_spans:
        if (src_beg, src_end) in src2tgt_token_mapping:
            continue
        tgt_beg = torch.argmax(sample_layer_attention_pooled[:, src_beg]).item()
        tgt_end = torch.argmax(sample_layer_attention_pooled[:, src_end]).item()

        # switch tgt begin and end indices
        if tgt_beg > tgt_end:
            tgt_beg, tgt_end = tgt_end, tgt_beg

        # whether to push begin or end of a span
        direction = 1
        topK = 1

        # try to find non overlapping spans to avoid overwriting old ones
        tgt_token_spans = list(src2tgt_token_mapping.values())
        while find_overlap(tgt_beg, tgt_end, tgt_token_spans) != -1:
            overlapping_span = tgt_token_spans[find_overlap(tgt_beg, tgt_end, tgt_token_spans)]
            # tail overlapping
            if tgt_beg < overlapping_span[0] and tgt_end < overlapping_span[1]:
                tgt_end = overlapping_span[0] - 1
            # head overlapping
            elif tgt_beg >= overlapping_span[0] and tgt_end > overlapping_span[1]:
                tgt_beg = overlapping_span[1] + 1
            # full span overlapping
            else:
                # find next best match
                if direction == 1:
                    topK += 1
                else:
                    direction *= -1
                if topK >= sample_layer_attention_pooled.size(0):
                    logger.error(f'Alignment failed for src_string: [{src_string}] and tgt_string: [{tgt_string}]')
                    break
                if direction == 1:
                    tgt_beg = torch.topk(sample_layer_attention_pooled[:, src_beg], topK).indices[-1].item()
                else:
                    tgt_end = torch.topk(sample_layer_attention_pooled[:, src_end], topK).indices[-1].item()

            # switch tgt begin and end indices
            if tgt_beg > tgt_end:
                tgt_beg, tgt_end = tgt_end, tgt_beg

        src2tgt_token_mapping[(src_beg, src_end)] = (tgt_beg, tgt_end)

    # create word mapping from token mapping between source and target sentences
    src2tgt_word_mapping = OrderedDict()
    for key, value in src2tgt_token_mapping.items():
        src_beg, src_end = src_token2word_mapping[key[0]], src_token2word_mapping[key[1]]
        tgt_beg, tgt_end = tgt_token2word_mapping[value[0]], tgt_token2word_mapping[value[1]]
        src2tgt_word_mapping[(src_beg, src_end)] = (tgt_beg, tgt_end)

    # sort src2tgt_word_mapping based on start of target spans
    src2tgt_word_mapping = dict(sorted(src2tgt_word_mapping.items(), key=lambda kv: kv[1][0]))

    output_tokens = []
    curr = 0
    for src_span, tgt_span in src2tgt_word_mapping.items():
        src_start, src_end = src_span
        tgt_start, tgt_end = tgt_span
        if tgt_start > curr:
            output_tokens.extend(tgt_words[curr:tgt_start])
        # +1 since it's inclusive
        replacement = ' '.join(src_words[src_start : src_end + 1])
        if remove_output_quotation:
            output_tokens.append(replacement)
        else:
            output_tokens.append(src_quotation_symbol + ' ' + replacement + ' ' + src_quotation_symbol)
        # +1 since it's inclusive
        curr = tgt_end + 1
    if curr < len(tgt_words):
        output_tokens.extend(tgt_words[curr:])

    output_string = ' '.join(output_tokens)

    return output_string
