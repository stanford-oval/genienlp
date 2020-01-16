#
# Copyright (c) 2018, The Board of Trustees of the Leland Stanford Junior University
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

from ..text.vocab import Vocab
from .example import Example, SequentialField


class DecoderVocabulary(object):
    def __init__(self, words, full_vocab):
        self.full_vocab = full_vocab
        if words is not None:
            self.itos = words
            self.stoi = { word: idx for idx, word in enumerate(words) }
        else:
            self.itos = []
            self.stoi = dict()
        self.oov_itos = []
        self.oov_stoi = dict()

    @property
    def max_generative_vocab(self):
        return len(self.itos)

    def clone(self):
        new_subset = DecoderVocabulary(None, self.full_vocab)
        new_subset.itos = self.itos
        new_subset.stoi = self.stoi
        return new_subset

    def __len__(self):
        return len(self.itos) + len(self.oov_itos)

    def encode(self, word):
        if word in self.stoi:
            lim_idx = self.stoi[word]
        elif word in self.oov_stoi:
            lim_idx = self.oov_stoi[word]
        else:
            lim_idx = len(self)
            self.oov_itos.append(word)
            self.oov_stoi[word] = lim_idx
        return lim_idx

    def decode(self, lim_idx):
        if lim_idx < len(self.itos):
            return self.full_vocab.stoi[self.itos[lim_idx]]
        else:
            return self.full_vocab.stoi[self.oov_itos[lim_idx-len(self.itos)]]


class SimpleNumericalizer(object):
    def __init__(self, max_effective_vocab, max_generative_vocab, fix_length=None, pad_first=False):
        self.max_effective_vocab = max_effective_vocab
        self.max_generative_vocab = max_generative_vocab

        self.init_token = '<init>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'

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

    def build_vocab(self, vectors, vocab_sets):
        self.vocab = Vocab.build_from_data(Example.vocab_fields, *vocab_sets,
                                           unk_token=self.unk_token,
                                           init_token=self.init_token,
                                           eos_token=self.eos_token,
                                           pad_token=self.pad_token,
                                           max_size=self.max_effective_vocab,
                                           vectors=vectors)
        self._init_vocab()

    def _grow_vocab_one(self, sentence, vectors, new_vectors):
        assert isinstance(sentence, list)

        # check if all the words are in the vocabulary, and if not
        # grow the vocabulary and the embedding matrix
        for word in sentence:
            if word not in self.vocab.stoi:
                self.vocab.stoi[word] = len(self.vocab.itos)
                self.vocab.itos.append(word)

                new_vector = [vec[word] for vec in vectors]

                # charNgram returns  a [1, D] tensor, while Glove returns a [D] tensor
                # normalize to [1, D] so we can concat along the second dimension
                # and later concat all vectors along the first
                new_vector = [vec if vec.dim() > 1 else vec.unsqueeze(0) for vec in new_vector]
                new_vectors.append(torch.cat(new_vector, dim=1))

    def grow_vocab(self, examples, vectors):
        new_vectors = []
        for ex in examples:
            self._grow_vocab_one(ex.context, vectors, new_vectors)
            self._grow_vocab_one(ex.question, vectors, new_vectors)
            self._grow_vocab_one(ex.answer, vectors, new_vectors)
        return new_vectors

    def _init_vocab(self):
        self.init_id = self.vocab.stoi[self.init_token]
        self.eos_id = self.vocab.stoi[self.eos_token]
        self.unk_id = self.vocab.stoi[self.unk_token]
        self.pad_id = self.vocab.stoi[self.pad_token]
        self.generative_vocab_size = min(self.max_generative_vocab, len(self.vocab))

        assert self.init_id < self.max_generative_vocab
        assert self.eos_id < self.max_generative_vocab
        assert self.unk_id < self.max_generative_vocab
        assert self.pad_id < self.max_generative_vocab

        self.decoder_vocab = DecoderVocabulary(self.vocab.itos[:self.max_generative_vocab], self.vocab)

    def encode(self, minibatch, decoder_vocab, device=None):
        if not isinstance(minibatch, list):
            minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded = []
        lengths = []
        numerical = []
        decoder_numerical = []
        for example in minibatch:
            if self.pad_first:
                padded_example = [self.pad_token] * max(0, max_len - len(example)) + \
                    ([] if self.init_token is None else [self.init_token]) + \
                    list(example[:max_len]) + \
                    ([] if self.eos_token is None else [self.eos_token])
            else:
                padded_example = ([] if self.init_token is None else [self.init_token]) + \
                                 list(example[:max_len]) + \
                                 ([] if self.eos_token is None else [self.eos_token]) + \
                                 [self.pad_token] * max(0, max_len - len(example))

            padded.append(padded_example)
            lengths.append(len(padded_example) - max(0, max_len - len(example)))

            numerical.append([self.vocab.stoi[word] for word in padded_example])
            decoder_numerical.append([decoder_vocab.encode(word) for word in padded_example])

        length = torch.tensor(lengths, dtype=torch.int32, device=device)
        numerical = torch.tensor(numerical, dtype=torch.int64, device=device)
        decoder_numerical = torch.tensor(decoder_numerical, dtype=torch.int64, device=device)

        return SequentialField(tokens=padded, length=length, value=numerical, limited=decoder_numerical)

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