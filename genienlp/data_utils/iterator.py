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

from genienlp.data_utils.example import NumericalizedExamples

logger = logging.getLogger(__name__)


class LengthSortedIterator:

    def __init__(
        self,
        dataset,
        batch_size,
    ):
        """
        batch_size: number of examples
        """
        self.dataset = dataset
        self.batch_size = batch_size  # number of examples
        self.length = len(dataset)
        self.last_batch_start_index = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.last_batch_start_index == self.length:
            raise StopIteration
        batch_of_indices = list(
            range(self.last_batch_start_index, min(self.last_batch_start_index + self.batch_size, self.length))
        )
        self.last_batch_start_index += len(batch_of_indices)
        return NumericalizedExamples.collate_batches(NumericalizedExamples.from_examples([self.dataset[i] for i in batch_of_indices]))
