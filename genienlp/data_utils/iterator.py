#
# Copyright (c) 2018, Salesforce, Inc.
#                     The Board of Trustees of the Leland Stanford Junior University
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

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

_warned_for_batch_size = False


class LengthSortedIterator(torch.utils.data.Sampler):
    """ """

    def __init__(
        self,
        data_source,
        batch_size,
        batch_size_fn,
        batching_algorithm='sample',
    ):
        """
        batch_size: can be number of tokens or number of examples, the type is inferred from batch_size_fn
        """
        self.batch_size_fn = batch_size_fn
        if batching_algorithm not in ['sample', 'epoch']:
            raise ValueError('--batching_algorithm must be one of `sample` or `epoch`')
        self.batching_algorithm = batching_algorithm

        self.data_source, self.original_order = data_source, list(range(len(data_source)))
        self.data_source_marked = np.zeros(shape=(len(self.data_source)))  # mark each example that has been used in a batch
        self.batch_size = batch_size  # number of examples or number of tokens
        self.last_batch_start_index = 0
        self.last_batch_start_index = self.last_batch_start_index

        # do not allow skipping examples during validation/ prediction
        self.no_skip = True
        # quickly iterate over self to calculate length
        self.length = 0
        for _ in self:
            self.length += 1
        # reset state
        self.last_batch_start_index = 0
        self.last_batch_start_index = self.last_batch_start_index
       

    def __len__(self):
        return self.length

    def __iter__(self):
        self.last_batch_start_index = 0
        self.data_source_marked = np.zeros(shape=(len(self.data_source)))
        self.last_batch_start_index = self.last_batch_start_index
        return self

    def __next__(self):
        batch_of_indices = []
        current_batch_size = 0
        candidate_index = self.last_batch_start_index
        if candidate_index >= len(self.data_source):
            # This is the end of the iterator
            raise StopIteration
        while current_batch_size < self.batch_size:
            candidate_example = self.data_source[candidate_index]
            if self.batch_size_fn([candidate_example]) > self.batch_size:
                # the example is too big even on its own
                global _warned_for_batch_size
                if self.no_skip:
                    raise ValueError(
                        'Have to skip examples in validation/ prediction splits. Increase the validation batch size'
                    )
                if not _warned_for_batch_size:
                    logger.warning(
                        'Skipping an example larger than batch size. Consider increasing the batch size to avoid this warning'
                    )
                    _warned_for_batch_size = True
                self.last_batch_start_index = self._next_unmarked_index(self.last_batch_start_index)
                candidate_index = self._next_unmarked_index(candidate_index)
                if candidate_index >= len(self.data_source):
                    raise StopIteration
                continue

            candidate_batch_size = self.batch_size_fn(
                [self.data_source[i] for i in batch_of_indices] + [candidate_example]
            )  # the new batch size if we added this example to the batch
            if candidate_batch_size > self.batch_size:
                # the new example would put us over the batch size limit
                break

            batch_of_indices.append(candidate_index)
            if self.batching_algorithm == 'epoch':
                self.data_source_marked[candidate_index] = 1  # mark this index until the end of this epoch
            current_batch_size = candidate_batch_size
            candidate_index = self._next_unmarked_index(candidate_index)

            if candidate_index == len(self.data_source):
                break  # don't start from i=0; there is a large difference between the length of the first and last element

        self.last_batch_start_index += len(batch_of_indices)
        return batch_of_indices

    def _unmarked_index_to_datasource_index(self, index: int) -> int:
        return np.searchsorted(np.arange(0, len(self.data_source)) - np.cumsum(self.data_source_marked), index, side='left')

    def _next_unmarked_index(self, index: int) -> int:
        """
        or stop at len(self.data_source)
        """
        index += 1
        while index < len(self.data_source) and self.data_source_marked[index] == 1:
            index += 1
        return index
