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

from .decoder_vocab import DecoderVocabulary
from .example import SequentialField

logger = logging.getLogger(__name__)



class TransformerNumericalizer(object):
    """
    Numericalizer that uses Tokenizers from huggingface's transformers library.
    """

    _special_tokens_to_word_map: List[Tuple[str, str]]
    _special_tokens_to_word_regexes: List[Tuple[re.Pattern, str]]
    _words_to_special_token_regexes: List[Tuple[re.Pattern, str]]

    def __init__(
        self, pretrained_tokenizer, args, save_dir=None
    ):
        """
        If `save_dir` is None, initializes a new Numericalizer and optionally adds new words to its vocabulary, otherwise,
        loads from `save_dir`
        """
        self._pretrained_name = pretrained_tokenizer
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
        self._init_decoder_vocab()

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
        else:
            tokenizer_args.update({'pretrained_model_name_or_path': self._pretrained_name})

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

        # pad_id for answer tokens (different than context/ question pad_id for token classification tasks)
        self.answer_pad_id = self.pad_id

    def _init_decoder_vocab(self):
        self.generative_vocab_size = len(self._tokenizer)
        self.decoder_vocab = None

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

    def process_classification_labels(self, all_context_plus_questions, all_answers):
        def tokenize_and_align_labels(all_sequences, all_sequences_wo_types, all_labels):
            tokenized_inputs = self._tokenizer.batch_encode_plus(
                all_sequences,
                padding=False,
                truncation=True,
                is_split_into_words=True,
            )
            tokenized_inputs_wo_types = self._tokenizer.batch_encode_plus(
                all_sequences_wo_types,
                padding=False,
                truncation=True,
                is_split_into_words=True,
            )

            all_processed_labels = []
            for i, label in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                word_ids_wo_types = tokenized_inputs_wo_types.word_ids(batch_index=i)

                # find feature size
                size_diff = len(word_ids) - len(word_ids_wo_types)

                previous_word_idx = None
                label_ids = []
                for word_id in word_ids_wo_types:
                    # special tokens's word_id is None
                    if word_id is None:
                        label_ids.append(self.answer_pad_id)
                    # set the label for the first subword of each word
                    elif word_id != previous_word_idx:
                        label_ids.append(label[word_id])
                    # process next subwords
                    else:
                        label_ids.append(label[word_id])
                    previous_word_idx = word_id

                # extend labels for features
                label_ids.extend([self.answer_pad_id] * size_diff)

                all_processed_labels.append(label_ids)

            return all_processed_labels

        all_context_plus_questions_wo_types = all_context_plus_questions

        assert all(
            [
                len(cpq.split(" ")) == len(answer.split(" "))
                for cpq, answer in zip(all_context_plus_questions_wo_types, all_answers)
            ]
        )

        # align labels
        tokenized_answers = tokenize_and_align_labels(
            [cpq.split(" ") for cpq in all_context_plus_questions],
            [cpq.split(" ") for cpq in all_context_plus_questions_wo_types],
            [list(map(lambda token: int(token), ans.split(" "))) for ans in all_answers],
        )

        batch_decoder_numerical = []
        if self.decoder_vocab:
            for i in range(len(tokenized_answers)):
                batch_decoder_numerical.append(self.decoder_vocab.encode(tokenized_answers[i]))
        else:
            batch_decoder_numerical = [[]] * len(tokenized_answers)

        answer_sequential_fields = []
        for i in range(len(tokenized_answers)):
            answer_sequential_fields.append(
                SequentialField(
                    value=tokenized_answers[i],
                    length=len(tokenized_answers[i]),
                    limited=batch_decoder_numerical[i],
                    feature=None,
                )
            )

        return answer_sequential_fields

    def encode_batch(self, sentences: List[str], field_name, features=None) -> List[SequentialField]:
        """
        Batched version of `encode_single()`. Uses multiprocessing on all CPU cores for preprocessing
        Inputs:
            sentences: a list of sentences to encode
            field_name: text field name (options: context, question, answer)
            features: for each sentence we have a list of features per token (used for NED)
        """
        # We need to set this so that `tokenizers` package does not complain about detecting forks.
        os.environ['TOKENIZERS_PARALLELISM'] = "true"

        if features is None:
            features = []
            extract_word_pieces = False
        else:
            assert all([len(sentence.split()) == len(feature) for sentence, feature in zip(sentences, features)])
            extract_word_pieces = True

        batch_size = len(sentences)

        if field_name != 'answer':
            sentences = [sent for sent in sentences]

        if self._preprocess_special_tokens:
            sentences, index2expansions = list(
                zip(
                    *map(
                        functools.partial(self._apply_special_token_preprocessing, return_idx2exp=bool(len(features))),
                        sentences,
                    )
                )
            )

            all_input_features = []
            if features:
                for i, (sentence, index2expansion) in enumerate(zip(sentences, index2expansions)):
                    feat = features[i]
                    new_feat = []
                    keys = set(index2expansion.keys())
                    for j in range(len(feat)):
                        repeat = 1
                        if j in keys:
                            repeat = index2expansion[j]

                        new_feat.extend(feat[j] * repeat)

                    assert len(new_feat) == len(sentence.split(' '))

                    all_input_features.append(new_feat)

            features = all_input_features

        def do_slow_tokenization(extract_word_pieces):
            all_input_ids = []
            all_wp_tokenized = []
            if extract_word_pieces:
                for i in range(batch_size):
                    text = sentences[i]
                    wp_tokenized = self._tokenizer.tokenize(text)
                    all_wp_tokenized.append(wp_tokenized)

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
            return batch_encoded, all_wp_tokenized

        if field_name == 'answer':
            with self._tokenizer.as_target_tokenizer():
                batch_encoded, all_wp_tokenized = do_slow_tokenization(extract_word_pieces)
        else:
            batch_encoded, all_wp_tokenized = do_slow_tokenization(extract_word_pieces)

        all_input_features = []

        if features:
            for i in range(batch_size):
                wp_features = []
                wp_tokenized = all_wp_tokenized[i]

                feat = features[i]

                # first token is always not a piece
                is_wp = [0] + [int(self._tokenizer.is_piece_fn(wp)) for wp in wp_tokenized[1:]]
                k = -1
                for j, wp in enumerate(wp_tokenized):
                    if not is_wp[j]:
                        k += 1
                    wp_features.append(feat[k])

                assert len(wp_tokenized) == len(wp_features)

                all_input_features.append(wp_features)

        features = all_input_features

        batch_special_tokens_mask = batch_encoded.special_tokens_mask

        batch_features = []

        if features:
            for i in range(batch_size):
                feat = features[i]
                special_tokens_mask = batch_special_tokens_mask[i]
                num_prefix_special_tokens, num_suffix_special_tokens = self.get_num_special_tokens(special_tokens_mask)

                pad_feat = Entity.get_pad_entity(self.args.max_features_size)
                feat = [pad_feat] * num_prefix_special_tokens + feat + [pad_feat] * num_suffix_special_tokens

                batch_features.append(feat)

        batch_numerical = batch_encoded.input_ids
        batch_length = batch_encoded.length

        batch_decoder_numerical = []
        if self.decoder_vocab:
            for i in range(batch_size):
                batch_decoder_numerical.append(self.decoder_vocab.encode(batch_numerical[i]))
        else:
            batch_decoder_numerical = [[]] * len(batch_numerical)

        sequential_fields = []
        for i in range(batch_size):
            if features:
                feature = [feat.flatten() for feat in batch_features[i]]
                assert len(batch_numerical[i]) == len(feature)
            else:
                feature = None

            sequential_fields.append(
                SequentialField(
                    value=batch_numerical[i],
                    length=batch_length[i],
                    limited=batch_decoder_numerical[i],
                    feature=feature,
                )
            )
        return sequential_fields

    def _apply_special_token_preprocessing(self, sentence, return_idx2exp=False):
        index2expansion = {}
        if return_idx2exp:
            for i, (regex, replacement) in enumerate(self._special_tokens_to_word_regexes):
                for match in regex.finditer(sentence):
                    idx = match.span()[0]
                    tokens_before_idx = len(sentence[:idx].split(' '))
                    index2expansion[tokens_before_idx] = len(replacement.split(' '))
        for i, (regex, replacement) in enumerate(self._special_tokens_to_word_regexes):
            sentence = regex.sub(replacement, sentence)
        # '^' is an unknown token to T5 tokenizer and will break the preprocessing.
        # '~' is also unknown to T5. Evaluating models in server mode will give wrong results since answers will not
        # go through genienlp and remain intact while predictions will be missing these tokens. We replace such tokens
        # with known ones that do not conflict with other tokens. This continues our series of
        # "Possible bugs in spm-based tokenizers" issued here https://github.com/huggingface/transformers/issues/12867
        return sentence, index2expansion

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
