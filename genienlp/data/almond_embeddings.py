#
# Copyright (c) 2018-2019, Salesforce, Inc.
#                          The Board of Trustees of the Leland Stanford Junior University
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

import torch

from .word_vectors import Vectors

ENTITIES = ['DATE', 'DURATION', 'EMAIL_ADDRESS', 'HASHTAG',
            'LOCATION', 'NUMBER', 'PHONE_NUMBER', 'QUOTED_STRING',
            'TIME', 'URL', 'USERNAME', 'PATH_NAME', 'CURRENCY']
MAX_ARG_VALUES = 5


class AlmondEmbeddings(Vectors):
    def __init__(self, name=None, cache=None, **kw):
        super().__init__(name, cache, **kw)

    def cache(self, name, cache, url=None):
        del name
        del cache

        dim = len(ENTITIES) + MAX_ARG_VALUES

        itos = []
        vectors = []
        for i, entity in enumerate(ENTITIES):
            for j in range(MAX_ARG_VALUES):
                itos.append(entity + '_' + str(j))
                vec = torch.zeros((dim,), dtype=torch.float32)
                vec[i] = 1.0
                vec[len(ENTITIES) + j] = 1.0
                vectors.append(vec)

        self.itos = itos
        self.stoi = {word: i for i, word in enumerate(itos)}
        self.vectors = torch.stack(vectors, dim=0).view(-1, dim)
        self.dim = dim
