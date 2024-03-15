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

import json
import logging
import os

from ..data_utils.example import Example
from .base_dataset import Dataset, Split

logger = logging.getLogger(__name__)


def make_example_id(dataset, example_id):
    return dataset.name + '/' + str(example_id)



def id_value(ex):
    id_ = ex.example_id.rsplit('/', 1)
    id_ = id_[0] if len(id_) == 1 else id_[1]
    return id_



class CQA(Dataset):
    def __init__(self, examples, **kwargs):
        super().__init__(examples, **kwargs)


class JSON(CQA):
    name = 'json'

    def __init__(self, path, subsample=None, **kwargs):

        examples = []
        with open(os.path.expanduser(path)) as f:
            lines = f.readlines()
            for line in lines:
                ex = json.loads(line)
                context, question, answer = ex['context'], ex['question'], ex['answer']
                examples.append(Example.from_raw(make_example_id(self, len(examples)), context, question, answer))
                if subsample is not None and len(examples) >= subsample:
                    break

        super(JSON, self).__init__(examples, **kwargs)

    @classmethod
    def splits(cls, root='.data', name=None, train='train', validation='val', test='test', **kwargs):
        path = os.path.join(root, name)

        train_data = None if train is None else cls(os.path.join(path, 'train.jsonl'), **kwargs)
        validation_data = None if validation is None else cls(os.path.join(path, 'val.jsonl'), **kwargs)
        test_data = None if test is None else cls(os.path.join(path, 'test.jsonl'), **kwargs)

        return Split(
            train=None if train is None else train_data,
            eval=None if validation is None else validation_data,
            test=None if test is None else test_data,
        )