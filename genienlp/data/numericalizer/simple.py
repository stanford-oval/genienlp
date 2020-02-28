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
import torch

from .vocab import Vocab
from .sequential_field import SequentialField
from .decoder_vocab import DecoderVocabulary


class SimpleNumericalizer(object):
    def __init__(self, max_generative_vocab, fix_length=None, pad_first=False):
        self.max_generative_vocab = max_generative_vocab

        self.init_token = '<init>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.mask_token = '<mask>'

        self.fix_length = fix_length
        self.pad_first = pad_first

    @property
    def num_tokens(self):
        return len(self.vocab)

    def load(self, save_dir):
        self.vocab = torch.load(os.path.join(save_dir, 'vocab.pth'))
        self._init_vocab()

    def save(self, save_dir):
        torch.save(self.vocab, os.path.join(save_dir, 'vocab.pth'))

    def build_vocab(self, vocab_fields, vocab_sets):
        self.vocab = Vocab.build_from_data(vocab_fields, *vocab_sets,
                                           unk_token=self.unk_token,
                                           init_token=self.init_token,
                                           eos_token=self.eos_token,
                                           pad_token=self.pad_token)
        self._init_vocab()

    def _grow_vocab_one(self, sentence, new_words):
        assert isinstance(sentence, list)

        # check if all the words are in the vocabulary, and if not
        # grow the vocabulary and the embedding matrix
        for word in sentence:
            if word not in self.vocab.stoi:
                self.vocab.stoi[word] = len(self.vocab.itos)
                self.vocab.itos.append(word)
                new_words.append(word)

    def grow_vocab(self, examples):
        new_words = []
        for ex in examples:
            self._grow_vocab_one(ex.context, new_words)
            self._grow_vocab_one(ex.question, new_words)
            self._grow_vocab_one(ex.answer, new_words)
        return new_words

    def _init_vocab(self):
        self.init_id = self.vocab.stoi[self.init_token]
        self.eos_id = self.vocab.stoi[self.eos_token]
        self.unk_id = self.vocab.stoi[self.unk_token]
        self.pad_id = self.vocab.stoi[self.pad_token]
        self.mask_id = self.vocab.stoi[self.mask_token]
        self.generative_vocab_size = min(self.max_generative_vocab, len(self.vocab))

        assert self.init_id < self.max_generative_vocab
        assert self.eos_id < self.max_generative_vocab
        assert self.unk_id < self.max_generative_vocab
        assert self.pad_id < self.max_generative_vocab
        assert self.mask_id < self.max_generative_vocab

        self.decoder_vocab = DecoderVocabulary(self.vocab.itos[:self.max_generative_vocab], self.vocab,
                                               pad_token=self.pad_token, eos_token=self.eos_token)

    def get_special_token_mask(self, tensor):
        special_tokens_tuple = (self.init_id, self.eos_id, self.pad_id, self.mask_id)
        return list(map(lambda x: 1 if x in special_tokens_tuple else 0, tensor))

    def encode(self, minibatch, decoder_vocab, device=None):
        assert isinstance(minibatch, list)
        if self.fix_length is None:
            max_len = max(len(x[0]) for x in minibatch)
        else:
            max_len = self.fix_length
        padded = []
        lengths = []
        numerical = []
        decoder_numerical = []
        for tokens, _mask in minibatch:
            if self.pad_first:
                padded_example = [self.pad_token] * max(0, max_len - len(tokens)) + \
                                 [self.init_token] + \
                                 list(tokens[:max_len]) + \
                                 [self.eos_token]
            else:
                padded_example = [self.init_token] + \
                                 list(tokens[:max_len]) + \
                                 [self.eos_token] + \
                                 [self.pad_token] * max(0, max_len - len(tokens))

            padded.append(padded_example)
            lengths.append(len(padded_example) - max(0, max_len - len(tokens)))

            numerical.append([self.vocab.stoi[word] for word in padded_example])
            decoder_numerical.append([decoder_vocab.encode(word) for word in padded_example])

        length = torch.tensor(lengths, dtype=torch.int32, device=device)
        numerical = torch.tensor(numerical, dtype=torch.int64, device=device)
        decoder_numerical = torch.tensor(decoder_numerical, dtype=torch.int64, device=device)

        return SequentialField(length=length, value=numerical, limited=decoder_numerical)

    def decode(self, tensor):
        return [self.vocab.itos[idx] for idx in tensor]

    def reverse(self, batch, detokenize, field_name=None):
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [self.decode(ex) for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        return [detokenize(ex, field_name=field_name) for ex in batch]
