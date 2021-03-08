#
# Copyright (c) 2019, The Board of Trustees of the Leland Stanford Junior University
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

from genienlp.tasks.almond_task import BaseAlmondTask

from ..tasks.registry import register_task
from ..data_utils.example import Example
from ..tasks.qa_dataset import AmbigQADataset, CONLLNERDataset

logger = logging.getLogger(__name__)


@register_task('ambig_qa')
class AmbigQA(BaseAlmondTask):

    def __init__(self, name, args):
        super().__init__(name, args)

    @property
    def metrics(self):
        return ['em', 'bleu']

    def _make_example(self, parts, dir_name=None, **kwargs):
        if len(parts) == 3:
            example_id, sentence, thingtalk = parts
        elif len(parts) == 4:
            example_id, _, sentence, thingtalk = parts # ignore dialogue context
        else:
            raise ValueError(f'Input file contains line with {len(parts)} parts: {str(parts)}')

        example_id = self.name + '/' + example_id

        question = 'translate from input to output'
        context = sentence
        answer = sentence # means we calculate self-bleu
        
        return Example.from_raw(example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)
    
    def get_splits(self, root, **kwargs):
        return AmbigQADataset.return_splits(path=root, make_example=self._make_example, **kwargs)


@register_task('conll_ner')
class CONLLNER(BaseAlmondTask):
    num_labels = 9
    label_list = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    
    def __init__(self, name, args):
        super().__init__(name, args)
    
    @property
    def metrics(self):
        return ['em', 'pem', 'f1']
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        if len(parts) == 3:
            example_id, sentence, thingtalk = parts
        elif len(parts) == 4:
            example_id, _, sentence, thingtalk = parts  # ignore dialogue context
        else:
            raise ValueError(f'Input file contains line with {len(parts)} parts: {str(parts)}')
        
        example_id = self.name + '/' + example_id
        
        question = 'translate from input to output'
        context = sentence
        answer = sentence  # means we calculate self-bleu
        
        return Example.from_raw(example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)
    
    def get_splits(self, root, **kwargs):
        return CONLLNERDataset.return_splits(path=root, make_example=self._make_example, **kwargs)

