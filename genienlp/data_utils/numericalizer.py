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

import os
import re
import json
import multiprocessing
from typing import List, Tuple
from collections import defaultdict, Counter
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer, BertTokenizer, XLMRobertaTokenizer

from .decoder_vocab import DecoderVocabulary
from .example import SequentialField, Feature, get_pad_feature

# not all tokenizers respect whitespace in the input or honor do_basic_tokenize=False
# for those, we need to use the slow tokenizers or we'll get messed up thingtalk output
from ..paraphrase.transformers_utils import SPIECE_UNDERLINE

ALLOWED_FAST_TOKENIZERS = {
    'facebook/bart-base',
    'facebook/bart-large',
    'sshleifer/bart-tiny-random'
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


class TransformerNumericalizer(object):
    """
    Numericalizer that uses Tokenizers from huggingface's transformers library.
    """
    
    _special_tokens_to_word_map: List[Tuple[str, str]]
    _special_tokens_to_word_regexes: List[Tuple[re.Pattern, str]]
    _special_tokens_to_token_regexes: List[Tuple[re.Pattern, str]]
    
    def __init__(self, pretrained_tokenizer, max_generative_vocab, cache=None, no_fast_tokenizer=False,
                 preprocess_special_tokens=False, features=None, features_default_val=None, features_size=None):
        self._pretrained_name = pretrained_tokenizer
        self.max_generative_vocab = max_generative_vocab
        self._cache = cache
        self._tokenizer = None
        
        self._preprocess_special_tokens = preprocess_special_tokens
        
        # map a token to a space-separated sequence of words
        self._special_tokens_to_word_map = []
        # same, but the token is a regular expression matching that token using \b
        self._special_tokens_to_word_regexes = []
        # map a space-separated sequence of words to a token
        self._special_tokens_to_token_regexes = []
        
        self.features = features
        self.features_default_val = features_default_val
        self.features_size = features_size
        
        self.no_fast_tokenizer = no_fast_tokenizer
        
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
        return not self.no_fast_tokenizer and \
               (self._pretrained_name in ALLOWED_FAST_TOKENIZERS or
               (self._preprocess_special_tokens and self._pretrained_name in ALLOWED_FAST_TOKENIZERS_IF_PREPROCESSING))
    

    def get_tokenizer(self, save_dir):
        if save_dir is not None:
            config = AutoConfig.from_pretrained(self._pretrained_name)
            self._tokenizer = AutoTokenizer.from_pretrained(save_dir,
                                                            do_lower_case=False,
                                                            do_basic_tokenize=False,
                                                            config=config,
                                                            cache_dir=self._cache,
                                                            use_fast=self._use_fast())
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(self._pretrained_name,
                                                            do_lower_case=False,
                                                            do_basic_tokenize=False,
                                                            cache_dir=self._cache,
                                                            use_fast=self._use_fast())
            
        if hasattr(self._tokenizer, 'wordpiece_tokenizer'):
            self._tokenizer.is_piece_fn = lambda wp: wp.startswith('##')
        else:
            self._tokenizer.is_piece_fn = lambda wp: not wp.startswith(SPIECE_UNDERLINE)


    def load(self, save_dir):
        if self.max_generative_vocab is not None:
            with open(os.path.join(save_dir, 'decoder-vocab.txt'), 'r') as fp:
                self._decoder_words = [(line.rstrip('\n'), self._tokenizer.convert_tokens_to_ids(line.rstrip('\n')))
                                       for line in fp]
        try:
            with open(os.path.join(save_dir, 'special-token-preprocessing.json')) as fp:
                self._special_tokens_to_word_map = json.load(fp)
            self._build_special_tokens_regexes()
        except FileNotFoundError:
            pass
        
        self._init()
    
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
            
            existing_special_tokens = self._tokenizer.special_tokens_map
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
            self._decoder_words = [(self._tokenizer.bos_token, self._tokenizer.bos_token_id),
                                   (self._tokenizer.eos_token, self._tokenizer.eos_token_id),
                                   (self._tokenizer.pad_token, self._tokenizer.pad_token_id),
                                   (self._tokenizer.unk_token, self._tokenizer.unk_token_id)] + \
                                  [(word, self._tokenizer.convert_tokens_to_ids(word)) for word, _freq
                                   in decoder_words.most_common(self.max_generative_vocab)]
        
        self._init()
    
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
                        assert len(word_sequences) < len(ambiguous_parts), (
                        original_token, ambiguous_token, word_sequences)
                        new_words = ambiguous_prefix + ' '.join(
                            ambiguous_parts[len(ambiguous_parts) - len(word_sequences) - 1:])
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
            word_re = re.compile("(^|(?<= ))" + re.escape(words) + "($|(?= ))")
            self._special_tokens_to_token_regexes.append((word_re, token))
            token_re = re.compile("(^|(?<= ))" + re.escape(token) + "(^|(?= ))")
            self._special_tokens_to_word_regexes.append((token_re, words))
    
    def _init(self):
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
        if self.max_generative_vocab is not None:
            self.generative_vocab_size = len(self._decoder_words)
            
            self.decoder_vocab = DecoderVocabulary(self._decoder_words, self._tokenizer,
                                                   pad_token=self.pad_token, eos_token=self.eos_token)
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
    
    def encode_batch(self, sentences: List[str], features: List[List[Feature]], multiprocessing_threshold=5000) -> List[
        SequentialField]:
        """
        Batched version of `encode_single()`. Uses multiprocessing on all CPU cores for preprocessing,
        and multithreading for tokenization if a `FastTokenizer` is used
        Inputs:
            sentences: a list of sentences to encode
            multiprocessing_threshold: for input batches smaller than this value, multiprocessing will not be used due to its overhead
        """
        # We need to set this so that `tokenizers` package does not complain about detecting forks.
        os.environ['TOKENIZERS_PARALLELISM'] = "true"
        
        batch_size = len(sentences)
        
        if self._preprocess_special_tokens:
            if len(sentences) > multiprocessing_threshold:
                with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                    sentences, index2expansions = list(zip(*map(self._apply_special_token_preprocessing, sentences)))
            else:
                sentences, index2expansions = list(zip(*map(self._apply_special_token_preprocessing, sentences)))

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
        # whereas slow version do not. We breakdwon slow tokenization into two steps
        # extract tokenized text first, use that to adjust features
        # then pass tokenized text to `_batch_prepare_for_model`
        all_input_ids = []
        all_wp_tokenized = []

        if self._use_fast():
            batch_encoded = self._tokenizer.batch_encode_plus(list(sentences),
                                                              add_special_tokens=True,
                                                              max_length=None,
                                                              return_length=True,
                                                              return_attention_mask=False,
                                                              return_special_tokens_mask=True
                                                              )

            
            for encoding in batch_encoded.encodings:
                # remove special tokens
                num_prefix_special_tokens, num_suffix_special_tokens = self.get_num_special_tokens(encoding.special_tokens_mask)
                wp_tokens = encoding.tokens[num_prefix_special_tokens: -num_suffix_special_tokens]
                all_wp_tokenized.append(wp_tokens)
            
        else:
            for i in range(batch_size):
                text = sentences[i]
                wp_tokenized = self._tokenizer.tokenize(text)
                all_wp_tokenized.append(wp_tokenized)
                
                # None indicates encoding single instance not paired inputs
                all_input_ids.append((self._tokenizer.convert_tokens_to_ids(wp_tokenized), None))

            batch_encoded = self._tokenizer._batch_prepare_for_model(all_input_ids,
                                                                     add_special_tokens=True,
                                                                     max_length=None,
                                                                     return_length=True,
                                                                     return_attention_mask=False,
                                                                     return_special_tokens_mask=True
                                                                     )

        
        all_input_features = []
        
        for i in range(batch_size):
            wp_features = []
            wp_tokenized = all_wp_tokenized[i]
        
            if features:
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
                
                pad_feat = get_pad_feature(self.features, self.features_default_val, self.features_size)
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
                SequentialField(value=batch_numerical[i], length=batch_length[i], limited=batch_decoder_numerical[i], feature=feature))
        return sequential_fields
    
    def _apply_special_token_preprocessing(self, sentence):
        index2expansion = {}
        for i, (regex, replacement) in enumerate(self._special_tokens_to_word_regexes):
            for match in regex.finditer(sentence):
                idx = match.span()[0]
                tokens_before_idx = len(sentence[:idx].split(' '))
                index2expansion[tokens_before_idx] = len(replacement.split(' '))
                
            sentence = regex.sub(replacement, sentence)
        return sentence, index2expansion
    
    def _undo_special_token_preprocessing(self, sentence):
        for regex, replacement in self._special_tokens_to_token_regexes:
            sentence = regex.sub(replacement, sentence)
        return sentence
    
    def reverse(self, batch, task=None, field_name=None):
        output = []
        for x in self._tokenizer.batch_decode(batch, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            if self._preprocess_special_tokens:
                x = self._undo_special_token_preprocessing(x)
            if task is not None and field_name == 'answer':
                x = task.postprocess_answer(x)
            output.append(x)
        
        return output
