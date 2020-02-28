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


class DecoderVocabulary(object):
    def __init__(self, words, full_vocab, pad_token, eos_token):
        self.full_vocab = full_vocab
        self.pad_token = pad_token
        self.eos_token = eos_token
        if words is not None:
            self.itos = words
            self.stoi = {word: idx for idx, word in enumerate(words)}
            self.pad_idx = self.stoi[pad_token]
            self.eos_idx = self.stoi[eos_token]
        else:
            self.itos = []
            self.stoi = dict()
            self.pad_idx = -1
            self.eos_idx = -1
        self.oov_itos = []
        self.oov_stoi = dict()

    def clone(self):
        new_subset = DecoderVocabulary(None, self.full_vocab, self.pad_token, self.eos_token)
        new_subset.itos = self.itos
        new_subset.stoi = self.stoi
        new_subset.pad_idx = self.stoi[self.pad_token]
        new_subset.eos_idx = self.stoi[self.eos_token]
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
            return self.full_vocab.stoi[self.oov_itos[lim_idx - len(self.itos)]]
