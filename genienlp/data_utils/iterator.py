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

import torch
import random

from .example import Batch
from ..tasks.generic_dataset import context_answer_len, default_batch_fn


class Iterator(torch.utils.data.IterableDataset):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 batch_size,
                 shuffle=False,
                 repeat=False,
                 use_data_batch_fn=False,
                 use_data_sort_key=False):
        # batch_size can be number of tokens or number of examples
        # the type is inferred from batch_size_fn
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        
        # used for sentence_batching
        self.groups = getattr(dataset, 'groups', None)
        
        if use_data_batch_fn:
            self.batch_size_fn = self.dataset.batch_size_fn
        else:
            self.batch_size_fn = default_batch_fn
        
        self.use_data_sort_key = use_data_sort_key
        if use_data_sort_key:
            self.sort_key = self.dataset.sort_key_fn
        else:
            self.sort_key = context_answer_len

    def __len__(self):
        if self.repeat:
            raise NotImplementedError()
        else:
            return len(self.dataset)

    def __iter__(self) -> Batch:
        while True:
            if self.shuffle:
                dataset = list(self.dataset)
                random.shuffle(dataset)
            else:
                dataset = self.dataset

            if self.use_data_sort_key:
                if self.groups:
                    batches = self._sentence_batching(dataset)
                else:
                    batches = self._bucket_batching(dataset)
            else:
                batches = self._batch(dataset, self.batch_size)

            for minibatch in batches:
                yield minibatch

            if not self.repeat:
                break

    def _batch(self, data, batch_size, fixed_size_only=False):
        """
        
        :param data:
        :param batch_size:
        :param fixed_size_only: only return batches with exactly batch_size number of samples;
        ** warning: only use this if you are passing actual length not number of tokens as batch_size.
        :return:
        """
        """Yield elements from data in chunks of batch_size.
        
        """
        minibatch = []
        size_so_far = 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch), size_so_far)
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                if fixed_size_only:
                    yield []
                if len(minibatch) == 1:  # if we only have one really big example
                    yield minibatch
                    minibatch, size_so_far = [], 0
                else:
                    yield minibatch[:-1]
                    minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1, 0)
                    if size_so_far > batch_size:  # if we add a really big example that needs to be on its own to a batch
                        yield minibatch
                        minibatch, size_so_far = [], 0
        if minibatch and not fixed_size_only:
            yield minibatch


    def _bucket_batching(self, data):
        """Sort within buckets, then batch, then shuffle batches.

        Partitions data into chunks of size 100*batch_size, sorts examples within
        each chunk using sort_key, then batch these examples and shuffle the
        batches.
        """
        for p in self._batch(data, self.batch_size * 100):
            p_batch = self._batch(sorted(p, key=self.sort_key), self.batch_size)
            if self.shuffle:
                p_batch = list(p_batch)
                random.shuffle(p_batch)
            for b in p_batch:
                yield b
                
    def _sentence_batching(self, data):
        """
        Sort the dataset using sort_key.
        Divide the batch into groups each representing minibatches of same sentences in different languages
        Shuffle order of minibatches within each minibatch.
        Regroup and return the batch
        """
        dataset_sorted = sorted(data, key=self.sort_key)
        for batch in self._batch(dataset_sorted, self.batch_size, fixed_size_only=True):
            assert self.batch_size % self.groups == 0
            
            if self.shuffle:
                minibatches = [batch[i: i+self.groups] for i in range(0, self.batch_size, self.groups)]
                random.shuffle(minibatches)
                # shuffle samples within each minibatch too
                # for minibatch in minibatches:
                #     random.shuffle(minibatch)
                batch = []
                for minibatch in minibatches:
                    batch.extend(minibatch)
            
            yield batch
            
