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
from collections import Counter, defaultdict
from typing import List, Tuple

from pathos import multiprocessing
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    SPIECE_UNDERLINE,
    T5_PRETRAINED_CONFIG_ARCHIVE_MAP,
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    ByT5Tokenizer,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    M2M100Tokenizer,
    MarianConfig,
    MarianTokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    T5Tokenizer,
    T5TokenizerFast,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
)

from ..util import get_devices
from .decoder_vocab import DecoderVocabulary
from .example import Entity, SequentialField

logger = logging.getLogger(__name__)

# not all tokenizers respect whitespace in the input or honor do_basic_tokenize=False
# for those, we need to use the slow tokenizers or we'll get messed up thingtalk output
ALLOWED_FAST_TOKENIZERS = {
    'facebook/bart-base',
    'facebook/bart-large',
    'sshleifer/bart-tiny-random',
}

# known NOT to work:
# - all the BERT models
# - all the XLM-R models
#
# mBART, t5, and mt5 models work when preprocessing, because they're SPM/RoBERTa-based so they respect
# whitespace, but the fast tokenizers treat special tokens differently than the slow ones
# and drop whitespace before special tokens, which breaks
ALLOWED_FAST_TOKENIZERS_IF_PREPROCESSING = {
    'facebook/mbart-large-cc25',
    'sshleifer/tiny-mbart',
    'google/mt5-small',
    'google/mt5-base',
    'google/mt5-large',
    'google/mt5-xl',
    'google/mt5-xxl',
}

# for input batches smaller than this value, multiprocessing will not be used due to its overhead
MULTIPROCESSING_THRESHOLD = 5000


class TransformerNumericalizer(object):
    """
    Numericalizer that uses Tokenizers from huggingface's transformers library.
    """

    _special_tokens_to_word_map: List[Tuple[str, str]]
    _special_tokens_to_word_regexes: List[Tuple[re.Pattern, str]]
    _words_to_special_token_regexes: List[Tuple[re.Pattern, str]]

    def __init__(
        self, pretrained_tokenizer, args, max_generative_vocab, config, src_lang, tgt_lang, vocab_sets, tasks, save_dir=None
    ):
        """
        If `save_dir` is None, initializes a new Numericalizer and optionally adds new words to its vocabulary, otherwise,
        loads from `save_dir`
        """
        self._pretrained_name = pretrained_tokenizer
        self.max_generative_vocab = max_generative_vocab
        self._cache = args.embeddings
        self._tokenizer = None
        self.config = config

        self._preprocess_special_tokens = args.preprocess_special_tokens

        # map a special token to a space-separated sequence of words
        self._special_tokens_to_word_map = []
        # same, but the token is a regular expression matching that token using \b
        self._special_tokens_to_word_regexes = []
        # map a space-separated sequence of words to a special token matching that sequence of words using regex
        self._words_to_special_token_regexes = []

        self.args = args

        self._init_tokenizer(save_dir, config, src_lang, tgt_lang)

        self.update_language_dependent_properties(src_lang, tgt_lang)

        if save_dir is not None:
            logger.info(f'Loading the accompanying numericalizer from {save_dir}')
            self.load_extras(save_dir)
        else:
            logger.info('Building vocabulary')
            self.build_vocab(vocab_sets, tasks)

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
        if self.max_generative_vocab is not None:
            return self.decoder_vocab.pad_idx
        else:
            return self.pad_id

    def _use_fast(self):
        if self.args.no_fast_tokenizer:
            return False
        if self.args.force_fast_tokenizer:
            return True
        return self._pretrained_name in ALLOWED_FAST_TOKENIZERS or (
            self._preprocess_special_tokens and self._pretrained_name in ALLOWED_FAST_TOKENIZERS_IF_PREPROCESSING
        )

    def _init_tokenizer(self, save_dir, config, src_lang, tgt_lang):
        """
        Initializes the `self._tokenizer` object, but not the rest.
        """
        tokenizer_args = {
            'do_lower_case': False,
            'do_basic_tokenize': False,
            'cache_dir': self._cache,
            'use_fast': self._use_fast(),
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
        }
        if save_dir is not None:
            tokenizer_args.update({'pretrained_model_name_or_path': save_dir, 'config': config})
        else:
            tokenizer_args.update({'pretrained_model_name_or_path': self._pretrained_name})

        self._tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)

        # We only include the base tokenizers since `isinstance` checks for inheritance
        if isinstance(self._tokenizer, (BertTokenizer, BertTokenizerFast)):
            self._tokenizer.is_piece_fn = lambda wp: wp.startswith('##')
        elif isinstance(
            self._tokenizer,
            (
                XLMRobertaTokenizer,
                XLMRobertaTokenizerFast,
                T5Tokenizer,
                T5TokenizerFast,
                MBart50Tokenizer,
                MBart50TokenizerFast,
                MarianTokenizer,
                M2M100Tokenizer,
            ),
        ):
            self._tokenizer.is_piece_fn = lambda wp: not wp.startswith(SPIECE_UNDERLINE)
        elif isinstance(self._tokenizer, (GPT2Tokenizer, GPT2TokenizerFast)):
            self._tokenizer.is_piece_fn = lambda wp: not wp.startswith('Ġ')
        elif isinstance(self._tokenizer, ByT5Tokenizer):
            self._tokenizer.is_piece_fn = lambda wp: False

        # make sure we assigned is_piece_fn
        assert self._tokenizer.is_piece_fn

    def update_language_dependent_properties(self, src_lang, tgt_lang):
        # some tokenizers like Mbart do not set src_lang and tgt_lan when initialized; take care of it here
        self._tokenizer.src_lang = src_lang
        self._tokenizer.tgt_lang = tgt_lang

        # define input prefix to add before every input text
        input_prefix = ''
        if isinstance(self.config, MarianConfig) and tgt_lang:
            input_prefix = f'>>{tgt_lang}<< '
        # only older T5 models need task-specific input prefix
        elif self._pretrained_name in T5_PRETRAINED_CONFIG_ARCHIVE_MAP.keys():
            assert src_lang == 'en'
            if tgt_lang == 'en':
                t5_task = 'summarization'
            else:
                t5_task = f'translation_en_to_{tgt_lang}'
            input_prefix = self.config.task_specific_params[t5_task]['prefix']

        self.input_prefix = input_prefix

    def load_extras(self, save_dir):
        if self.max_generative_vocab is not None:
            with open(os.path.join(save_dir, 'decoder-vocab.txt'), 'r') as fp:
                self._decoder_words = [
                    (line.rstrip('\n'), self._tokenizer.convert_tokens_to_ids(line.rstrip('\n'))) for line in fp
                ]
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

    def save(self, save_dir):
        self._tokenizer.save_pretrained(save_dir)
        if self.max_generative_vocab is not None:
            with open(os.path.join(save_dir, 'decoder-vocab.txt'), 'w') as fp:
                for word, _full_idx in self._decoder_words:
                    fp.write(word + '\n')
        if len(self._special_tokens_to_word_map) > 0:
            with open(os.path.join(save_dir, 'special-token-preprocessing.json'), 'w') as fp:
                json.dump(self._special_tokens_to_word_map, fp)

    def build_vocab(self, vocab_sets, tasks):
        special_tokens = []
        for task in tasks:
            special_tokens += list(task.special_tokens)
        special_tokens.sort()

        if self._preprocess_special_tokens:
            self._build_special_tokens_maps(special_tokens)
            self._build_special_tokens_regexes()
        else:
            # add the special tokens directly to the tokenizer
            self._tokenizer.add_tokens(special_tokens)

        # add entity boundary special tokens
        if self.args.add_entities_to_text != 'off':
            self._tokenizer.add_tokens(['<e>', '</e>'])

        # add special tokens for ambig_qa task
        if any(task.name == 'ambig_qa' for task in tasks):
            self._tokenizer.add_tokens(['<q>', '<p>', '<u>'])

        existing_special_tokens = self._tokenizer.special_tokens_map
        # add separator if it doesn't exist. It will be used to concatenate context and question
        if 'sep_token' not in existing_special_tokens:
            self._tokenizer.add_special_tokens({'sep_token': existing_special_tokens.get('sep_token', '</s>')})

        if self.max_generative_vocab is not None:
            # do a pass over all the data in the dataset
            # in this pass, we
            # 1) tokenize everything, to ensure we account for all added tokens
            # 2) we construct a counter of wordpieces in the answers, for the decoder vocabulary
            decoder_words = Counter()
            for dataset in vocab_sets:
                for example in dataset:
                    decoder_words.update(self._tokenizer.tokenize(example.context))
                    decoder_words.update(self._tokenizer.tokenize(example.question))
                    decoder_words.update(self._tokenizer.tokenize(example.answer))

            # add the required special tokens, if not present already
            # note: if the tokens are not present, it means they are not used natively
            # by the model, so we can pick our favorite token
            if 'bos_token' not in existing_special_tokens:
                self._tokenizer.add_special_tokens({'bos_token': existing_special_tokens.get('cls_token', '<s>')})
            if 'eos_token' not in existing_special_tokens:
                self._tokenizer.add_special_tokens({'eos_token': existing_special_tokens.get('sep_token', '</s>')})
            if 'pad_token' not in existing_special_tokens:
                self._tokenizer.add_special_tokens({'pad_token': '<pad>'})
            if 'unk_token' not in existing_special_tokens:
                self._tokenizer.add_special_tokens({'unk_token': '<unk>'})
            self._decoder_words = [
                (self._tokenizer.bos_token, self._tokenizer.bos_token_id),
                (self._tokenizer.eos_token, self._tokenizer.eos_token_id),
                (self._tokenizer.pad_token, self._tokenizer.pad_token_id),
                (self._tokenizer.unk_token, self._tokenizer.unk_token_id),
            ] + [
                (word, self._tokenizer.convert_tokens_to_ids(word))
                for word, _freq in decoder_words.most_common(self.max_generative_vocab)
            ]

    def grow_vocab(self, tasks):
        if self._preprocess_special_tokens:
            # if we're preprocessing special tokens, we cannot extend the vocabulary
            # if the vocabulary was incomplete during training, tough luck, those words will be subwords all the way
            # (what do you expect?)
            return

        # add the new special tokens from the task
        for task in tasks:
            self._tokenizer.add_tokens(list(task.special_tokens))

    def _build_special_tokens_maps(self, special_tokens):
        # we automatically construct the mapping from special tokens to the shortest unambiguous
        # sequence of word-like things

        processed_tokens = dict()
        # first, split each token into words
        for original_token in special_tokens:
            token = original_token
            prefix = None
            if token.startswith('@'):
                prefix = '@ '
                token = token[1:]
            elif token.startswith('^^'):
                prefix = '^^ '
                token = token[2:]

            # split the token into words
            parts = re.split('[:._-]', token)
            if not prefix:
                # if we don't have a prefix, use the first word as prefix
                assert len(parts) >= 2
                prefix = parts[0] + ' '
                parts = parts[1:]
            processed_tokens[original_token] = (prefix, parts)

        # words -> token(s) (multiple tokens if a sequence of words is ambiguous)
        assignment = defaultdict(list)
        # token -> words
        reverse_mapping = dict()

        # now greedily assign each token to the shortest end that is not ambiguous
        for original_token, (prefix, parts) in processed_tokens.items():
            for i in range(len(parts) - 1, -1, -1):
                attempt = prefix + ' '.join(parts[i:])
                if attempt in assignment:
                    # ambiguous, extend everything in the current assignment by one word
                    for ambiguous_token in assignment[attempt]:
                        # list of sequences of words mapping to this token
                        # the first one is the one we have chosen so far, and the others are ambiguous
                        word_sequences = reverse_mapping[ambiguous_token]
                        if word_sequences[0] != attempt:
                            # ambiguous_token is already choosing a word sequence that is not ambiguous with
                            # original_token, nothing to do, other than to know original_token will need to
                            # be extended
                            continue
                        # extend ambiguous_token by one
                        ambiguous_prefix, ambiguous_parts = processed_tokens[ambiguous_token]
                        # assert we still have one word to use to disambiguate
                        # this works as long as the tokens are not suffix of one another
                        assert len(word_sequences) < len(ambiguous_parts), (original_token, ambiguous_token, word_sequences)
                        new_words = ambiguous_prefix + ' '.join(
                            ambiguous_parts[len(ambiguous_parts) - len(word_sequences) - 1 :]
                        )
                        word_sequences.insert(0, new_words)
                        # before original_token, ambiguous_token was not ambiguous with any token already
                        # assigned, so it cannot be ambiguous after we made it longer
                        assert new_words not in assignment
                        assignment[new_words] = [ambiguous_token]

                    # mark that attempt is an ambiguous suffix of original_token
                    assignment[attempt].append(original_token)

                    # don't assign original_token at this step, wait until the next loop cycle
                    # at the next loop, we'll try again with a longer suffix of original_token
                    # that way, we check if the token is still ambiguous after we extended everything
                    # else by one word
                else:
                    # yay not ambiguous, time to assign it

                    # construct all word sequences, from the one we chose to the end
                    word_sequences = [prefix + ' '.join(parts[j:]) for j in range(i, len(parts))]
                    assignment[attempt] = [original_token]
                    reverse_mapping[original_token] = word_sequences
                    break

        # okay we have assigned everything, time to clean up
        for token, word_sequences in reverse_mapping.items():
            self._special_tokens_to_word_map.append((token, word_sequences[0]))

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
        if self.max_generative_vocab is not None:
            self.generative_vocab_size = len(self._decoder_words)

            self.decoder_vocab = DecoderVocabulary(
                self._decoder_words, self._tokenizer, pad_token=self.pad_token, eos_token=self.eos_token
            )
        else:
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

        if self.args.add_entities_to_text == 'insert':
            raise ValueError('Insert option for add_entities_to_text argument is not supported for token_classification tasks')
        elif self.args.add_entities_to_text == 'append':
            all_context_plus_questions_wo_types = [example[: example.index(' <e>')] for example in all_context_plus_questions]
        else:
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
        Batched version of `encode_single()`. Uses multiprocessing on all CPU cores for preprocessing,
        and multithreading for tokenization if a `FastTokenizer` is used
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
            sentences = [self.input_prefix + sent for sent in sentences]

        if self._preprocess_special_tokens:
            if len(sentences) > MULTIPROCESSING_THRESHOLD:
                multiprocessing_factor = multiprocessing.cpu_count() // len(get_devices(self.args.devices))
                logger.info('multiprocessing factor for special token preprocessing is %d', multiprocessing_factor)
                with multiprocessing.Pool(multiprocessing_factor) as p:
                    sentences, index2expansions = list(
                        zip(
                            *p.map(
                                functools.partial(self._apply_special_token_preprocessing, return_idx2exp=bool(len(features))),
                                sentences,
                            )
                        )
                    )
            else:
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

        # batch_encode_plus for fast tokenizers returns tokenized text
        # whereas slow version do not. We breakdown slow tokenization into two steps
        # extract tokenized text first, use that to adjust features
        # then pass tokenized text to `_batch_prepare_for_model`
        def do_fast_tokenization(extract_word_pieces):
            all_wp_tokenized = []
            batch_encoded = self._tokenizer.batch_encode_plus(
                list(sentences),
                add_special_tokens=True,
                max_length=None,
                return_length=True,
                return_attention_mask=False,
                return_special_tokens_mask=True,
            )
            if extract_word_pieces:
                for encoding in batch_encoded.encodings:
                    # remove special tokens
                    num_prefix_special_tokens, num_suffix_special_tokens = self.get_num_special_tokens(
                        encoding.special_tokens_mask
                    )
                    wp_tokens = encoding.tokens[num_prefix_special_tokens:-num_suffix_special_tokens]
                    all_wp_tokenized.append(wp_tokens)

            return batch_encoded, all_wp_tokenized

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

        if self._use_fast():
            if field_name == 'answer':
                with self._tokenizer.as_target_tokenizer():
                    batch_encoded, all_wp_tokenized = do_fast_tokenization(extract_word_pieces)
            else:
                batch_encoded, all_wp_tokenized = do_fast_tokenization(extract_word_pieces)

        else:
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
        if isinstance(self._tokenizer, (T5Tokenizer, T5TokenizerFast)):
            sentence = sentence.replace('^^', '%')
            sentence = sentence.replace('~', '#')
        return sentence, index2expansion

    def _undo_special_token_preprocessing(self, sentence):
        # undo T5 specific token preprocessing
        if isinstance(self._tokenizer, (T5Tokenizer, T5TokenizerFast)):
            sentence = sentence.replace('%', '^^')
            sentence = sentence.replace('#', '~')
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
