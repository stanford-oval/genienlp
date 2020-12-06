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

import collections
import os
from torch.nn.utils.rnn import pad_sequence

from .decoder_vocab import DecoderVocabulary
from transformers import AutoTokenizer
from .sequential_field import SequentialField

class TransformerNumericalizer(object):
    """
    Numericalizer that uses Tokenizers from huggingface's transformers library.
    """

    def __init__(self, pretrained_tokenizer, config, max_generative_vocab, cache=None, fix_length=None):
        self.config = config
        self._pretrained_name = pretrained_tokenizer
        self.max_generative_vocab = max_generative_vocab
        self._cache = cache
        self._tokenizer = None
        self.fix_length = fix_length

    @property
    def vocab(self):
        return self._tokenizer

    @property
    def num_tokens(self):
        return len(self._tokenizer)

    def load(self, save_dir):
        self._tokenizer = AutoTokenizer.from_pretrained(save_dir, config=self.config, cache_dir=self._cache)
        # HACK we cannot save the tokenizer without this
        del self._tokenizer.init_kwargs['config']

        if self.max_generative_vocab is not None:
            with open(os.path.join(save_dir, 'decoder-vocab.txt'), 'r') as fp:
                self._decoder_words = [line.rstrip('\n') for line in fp]

        self._init()

    def pad(self, batch):
        """
        batch: a List of List of integers
        """
        #TODO account for left padding models
        return pad_sequence(batch, padding_value=self.pad_id, batch_first=True)

    def save(self, save_dir):
        self._tokenizer.save_pretrained(save_dir)
        if self.max_generative_vocab is not None:
            with open(os.path.join(save_dir, 'decoder-vocab.txt'), 'w') as fp:
                for word in self._decoder_words:
                    fp.write(word + '\n')

    def build_vocab(self, vocab_sets, tasks):
        self._tokenizer = AutoTokenizer.from_pretrained(self._pretrained_name, config=self.config,
                                                        cache_dir=self._cache)
        # HACK we cannot save the tokenizer without this
        del self._tokenizer.init_kwargs['config']

        # ensure that init, eos, unk and pad are set
        # this method has no effect if the tokens are already set according to the tokenizer class
        self._tokenizer.add_special_tokens({
            'bos_token': "<s>",
            'eos_token': "</s>",
            'sep_token': "</s>",
            'unk_token': "<unk>",
            'pad_token': "<pad>",
            'mask_token': "<mask>",
            'cls_token': "<s>",
        })

        # add the special tokens from the task
        for task in tasks:
            self._tokenizer.add_tokens(task.special_tokens)

        if self.max_generative_vocab is not None:
            # do a pass over all the data in the dataset
            # in this pass, we
            # 1) tokenize everything, to ensure we account for all added tokens
            # 2) we construct a counter of wordpieces in the answers, for the decoder vocabulary
            decoder_words = collections.Counter()
            for dataset in vocab_sets:
                for example in dataset:
                    decoder_words.update(self._tokenizer.tokenize(example.context, example.context_word_mask))
                    decoder_words.update(self._tokenizer.tokenize(example.question, example.question_word_mask))
                    decoder_words.update(self._tokenizer.tokenize(example.answer, example.answer_word_mask))

            self._decoder_words = [self._tokenizer.bos_token, self._tokenizer.eos_token, self._tokenizer.pad_token,
                                   self._tokenizer.unk_token, self._tokenizer.mask_token] + \
                                  [word for word, _freq in decoder_words.most_common(self.max_generative_vocab)]

        self._init()

    def grow_vocab(self, tasks):
        # add the new special tokens from the task
        for task in tasks:
            self._tokenizer.add_tokens(task.special_tokens)

    def get_special_token_mask(self, token_ids):
        special_tokens_tuple = (self.init_id, self.eos_id, self.pad_id, self.mask_id)
        return list(map(lambda x: 1 if x in special_tokens_tuple else 0, token_ids))

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

    def encode_single(self, minibatch, decoder_vocab, max_length=-1):
        assert isinstance(minibatch, list)

        # apply word-piece tokenization to everything first
        wp_tokenized = []
        for field in minibatch:
            wp_tokenized.append(self._tokenizer.tokenize(field))

        if max_length > -1:
            max_len = max_length
        elif self.fix_length is None:
            max_len = max(len(x) for x in wp_tokenized)
        else:
            max_len = self.fix_length

        examples = []
        lengths = []
        numerical = []
        decoder_numerical = []
        for wp_tokens in wp_tokenized:
            example = [self.init_token] + list(wp_tokens[:max_len]) + [self.eos_token]

            examples.append(example)
            lengths.append(len(example))

            numerical.append(self._tokenizer.convert_tokens_to_ids(example))
            if decoder_vocab is not None:
                decoder_numerical.append([decoder_vocab.encode(word) for word in example])

        return SequentialField(length=lengths, value=numerical, limited=decoder_numerical)

    def decode(self, tensor):
        return self._tokenizer.convert_ids_to_tokens(tensor)

    def reverse(self, batch):
        raise self._tokenizer.batch_decode(batch)