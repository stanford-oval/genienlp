#
# Copyright (c) 2019-2020 The Board of Trustees of the Leland Stanford Junior University
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

import functools
import json
import logging
import os
import re
from typing import List, Tuple

from torch.nn.utils.rnn import pad_sequence
from transformers import (
    SPIECE_UNDERLINE,
    AutoTokenizer,
)

from .example import SequentialField

logger = logging.getLogger(__name__)


class TransformerNumericalizer(object):
    """
    Numericalizer that uses Tokenizers from huggingface's transformers library.
    """

    _special_tokens_to_word_map: List[Tuple[str, str]]
    _special_tokens_to_word_regexes: List[Tuple[re.Pattern, str]]
    _words_to_special_token_regexes: List[Tuple[re.Pattern, str]]

    def __init__(self, args, save_dir=None):
        """
        If `save_dir` is None, initializes a new Numericalizer and optionally adds new words to its vocabulary, otherwise,
        loads from `save_dir`
        """
        self._tokenizer = None

        self._preprocess_special_tokens = args.preprocess_special_tokens

        # map a special token to a space-separated sequence of words
        self._special_tokens_to_word_map = []
        # same, but the token is a regular expression matching that token using \b
        self._special_tokens_to_word_regexes = []
        # map a space-separated sequence of words to a special token matching that sequence of words using regex
        self._words_to_special_token_regexes = []

        self.args = args

        self._init_tokenizer(save_dir)

        if save_dir is not None:
            logger.info(f'Loading the accompanying numericalizer from {save_dir}')
            self.load_extras(save_dir)

        self._init_token_ids()

    @property
    def vocab(self):
        return self._tokenizer

    @property
    def num_tokens(self):
        return len(self._tokenizer)

    @property
    def decoder_pad_id(self):
        return self.pad_id

    def _init_tokenizer(self, save_dir):
        """
        Initializes the `self._tokenizer` object, but not the rest.
        """
        tokenizer_args = {
            'do_lower_case': False,
            'do_basic_tokenize': False,
        }
        if save_dir is not None:
            tokenizer_args.update({'pretrained_model_name_or_path': save_dir})

        self._tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)

        self._tokenizer.is_piece_fn = lambda wp: not wp.startswith(SPIECE_UNDERLINE)

        # make sure we assigned is_piece_fn
        assert self._tokenizer.is_piece_fn

    def load_extras(self, save_dir):
        try:
            with open(os.path.join(save_dir, 'special-token-preprocessing.json')) as fp:
                self._special_tokens_to_word_map = json.load(fp)
            self._build_special_tokens_regexes()
        except FileNotFoundError:
            pass

    def pad(self, batch, pad_id):
        """
        batch: a List of List of integers
        """
        # TODO account for left padding models
        return pad_sequence(batch, padding_value=pad_id, batch_first=True)

    def _build_special_tokens_regexes(self):
        for token, words in self._special_tokens_to_word_map:
            # match requiring (at the beginning of the string or preceded by a space (positive lookbehind))
            # and (at the end of the string or followed by a space (positive lookahead))
            token_re = re.compile("(^|(?<= ))" + re.escape(token) + "(^|(?= ))")
            self._special_tokens_to_word_regexes.append((token_re, words))
            word_re = re.compile("(^|(?<= ))" + re.escape(words) + "($|(?= ))")
            self._words_to_special_token_regexes.append((word_re, token))

    def _init_token_ids(self):
        self.pad_first = self._tokenizer.padding_side == 'left'

        self.init_token = self._tokenizer.bos_token
        self.eos_token = self._tokenizer.eos_token
        self.unk_token = self._tokenizer.unk_token
        self.pad_token = self._tokenizer.pad_token
        self.mask_token = self._tokenizer.mask_token
        self.cls_token = self._tokenizer.cls_token
        self.sep_token = self._tokenizer.sep_token

        self.init_id = self._tokenizer.bos_token_id
        self.eos_id = self._tokenizer.eos_token_id
        self.unk_id = self._tokenizer.unk_token_id
        self.pad_id = self._tokenizer.pad_token_id
        self.mask_id = self._tokenizer.mask_token_id
        self.cls_id = self._tokenizer.cls_token_id
        self.sep_id = self._tokenizer.sep_token_id

    def get_num_special_tokens(self, special_tokens_mask):
        num_prefix_special_tokens, num_suffix_special_tokens = 0, 0
        i = 0
        while i < len(special_tokens_mask):
            if special_tokens_mask[i] == 1:
                num_prefix_special_tokens += 1
                i += 1
            else:
                break
        i = len(special_tokens_mask) - 1
        while i >= 0:
            if special_tokens_mask[i] == 1:
                num_suffix_special_tokens += 1
                i -= 1
            else:
                break

        return num_prefix_special_tokens, num_suffix_special_tokens

    def encode_batch(self, sentences: List[str], field_name) -> List[SequentialField]:
        """
        Batched version of `encode_single()`. Uses multiprocessing on all CPU cores for preprocessing
        Inputs:
            sentences: a list of sentences to encode
            field_name: text field name (options: context, question, answer)
        """

        extract_word_pieces = False

        batch_size = len(sentences)

        if field_name != 'answer':
            sentences = [sent for sent in sentences]

        if self._preprocess_special_tokens:
            sentences = map(
                self._apply_special_token_preprocessing,
                sentences,
            )

        def do_slow_tokenization(extract_word_pieces):
            all_input_ids = []
            if extract_word_pieces:
                for i in range(batch_size):
                    text = sentences[i]
                    wp_tokenized = self._tokenizer.tokenize(text)

                    # None indicates encoding single instance not paired inputs
                    all_input_ids.append((self._tokenizer.convert_tokens_to_ids(wp_tokenized), None))

                batch_encoded = self._tokenizer._batch_prepare_for_model(
                    all_input_ids,
                    add_special_tokens=True,
                    max_length=None,
                    return_length=True,
                    return_attention_mask=False,
                    return_special_tokens_mask=True,
                )
            else:
                batch_encoded = self._tokenizer.batch_encode_plus(
                    list(sentences),
                    add_special_tokens=True,
                    max_length=None,
                    return_length=True,
                    return_attention_mask=False,
                    return_special_tokens_mask=True,
                )
            return batch_encoded

        if field_name == 'answer':
            with self._tokenizer.as_target_tokenizer():
                batch_encoded = do_slow_tokenization(extract_word_pieces)
        else:
            batch_encoded = do_slow_tokenization(extract_word_pieces)

        batch_numerical = batch_encoded.input_ids
        batch_length = batch_encoded.length

        batch_decoder_numerical = []
        batch_decoder_numerical = [[]] * len(batch_numerical)

        sequential_fields = []
        for i in range(batch_size):
            sequential_fields.append(
                SequentialField(
                    value=batch_numerical[i],
                    length=batch_length[i],
                    limited=batch_decoder_numerical[i],
                )
            )
        return sequential_fields

    def _apply_special_token_preprocessing(self, sentence):
        for regex, replacement in self._special_tokens_to_word_regexes:
            sentence = regex.sub(replacement, sentence)
        # '^' is an unknown token to T5 tokenizer and will break the preprocessing.
        # '~' is also unknown to T5. Evaluating models in server mode will give wrong results since answers will not
        # go through genienlp and remain intact while predictions will be missing these tokens. We replace such tokens
        # with known ones that do not conflict with other tokens. This continues our series of
        # "Possible bugs in spm-based tokenizers" issued here https://github.com/huggingface/transformers/issues/12867
        return sentence

    def _undo_special_token_preprocessing(self, sentence):
        # undo T5 specific token preprocessing
        for regex, replacement in self._words_to_special_token_regexes:
            sentence = regex.sub(replacement, sentence)
        return sentence

    def reverse(self, batch, field_name, skip_special_tokens=True):
        output = []
        use_source_tokenizer = True
        if field_name == 'answer':
            use_source_tokenizer = False
        for x in self._tokenizer.batch_decode(
            batch,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
            use_source_tokenizer=use_source_tokenizer,
        ):
            if self._preprocess_special_tokens:
                x = self._undo_special_token_preprocessing(x)
            output.append(x)
        return output

    def convert_ids_to_tokens(self, batch, skip_special_tokens):
        output = []
        for ids in batch:
            x = self._tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
            if self._preprocess_special_tokens:
                x = self._undo_special_token_preprocessing(x)
            output.append(x)
        return output
