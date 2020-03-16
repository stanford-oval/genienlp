#
# Copyright (c) 2018-2019 The Board of Trustees of the Leland Stanford Junior University
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

import numpy as np
from tqdm import tqdm


def string_hash(x):
    """ Simple deterministic string hash

    Based on https://cp-algorithms.com/string/string-hashing.html.
    We need this because str.__hash__ is not deterministic (it varies with each process restart)
    and it uses 8 bytes (which is too much for our uses)
    """

    P = 1009
    h = 0
    for c in x:
        h = (h << 10) + h + ord(c) * P
        h = h & 0xFFFFFFFF
    return np.uint32(h)


class HashTable(object):
    EMPTY_BUCKET = 0

    def __init__(self, itos, table=None):
        # open addressing hashing, with load factor 0.50

        if table is not None:
            assert isinstance(itos, np.ndarray)

            self.itos = itos
            self.table = table
            self.table_size = table.shape[0]
        else:

            max_str_len = max(len(x) for x in itos)
            self.itos = np.array(itos, dtype='U' + str(max_str_len))

            self.table_size = int(len(itos) * 2)
            self.table = np.zeros((self.table_size,), dtype=np.int64)

            self._build(itos)

    def _build(self, itos):
        for i, word in enumerate(tqdm(itos, total=len(itos))):
            hash = string_hash(word)
            bucket = hash % self.table_size

            while self.table[bucket] != self.EMPTY_BUCKET:
                hash += 7
                bucket = hash % self.table_size

            self.itos[i] = word
            self.table[bucket] = 1 + i

    def __iter__(self):
        return iter(self.itos)

    def __reversed__(self):
        return reversed(self.itos)

    def __len__(self):
        return self.itos

    def __eq__(self, other):
        return isinstance(other, HashTable) and self.itos == other.itos

    def __hash__(self):
        return hash(self.itos)

    def _find(self, key):
        hash = string_hash(key)
        for probe_count in range(self.table_size):
            bucket = (hash + 7 * probe_count) % self.table_size

            key_index = self.table[bucket]
            if key_index == self.EMPTY_BUCKET:
                return None

            if self.itos[key_index - 1] == key:
                return key_index - 1
        return None

    def __getitem__(self, key):
        found = self._find(key)
        if found is None:
            raise KeyError(f'Invalid key {key}')
        else:
            return found

    def __contains__(self, key):
        found = self._find(key)
        return found is not None

    def get(self, key, default=None):
        found = self._find(key)
        if found is None:
            return default
        else:
            return found
