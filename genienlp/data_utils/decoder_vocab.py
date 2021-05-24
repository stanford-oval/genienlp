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
        stoi = {word: idx for idx, (word, full_idx) in enumerate(words)}
        self.limited_to_full = {stoi[word]: full_idx for word, full_idx in words}
        self.full_to_limited = {full_idx: stoi[word] for word, full_idx in words}
        self.pad_idx = stoi[pad_token]
        self.eos_idx = stoi[eos_token]

    def __len__(self):
        return len(self.limited_to_full)

    def encode(self, full_idx_list):
        limited_list = []
        for full_idx in full_idx_list:
            if full_idx in self.full_to_limited:
                lim_idx = self.full_to_limited[full_idx]
            else:
                lim_idx = len(self)
                self.limited_to_full[lim_idx] = full_idx
                self.full_to_limited[full_idx] = lim_idx
            limited_list.append(lim_idx)
        return limited_list

    def decode(self, lim_idx):
        return self.limited_to_full[lim_idx]
