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

from .generic_dataset import CrossNERDataset
from .hf_dataset import HFDataset
from ..tasks.base_task import BaseTask
from ..tasks.registry import register_task
from ..data_utils.example import Example

logger = logging.getLogger(__name__)


class HFTask(BaseTask):
    def __init__(self, name, args):
        super().__init__(name, args)


@register_task('ambig_qa')
class AmbigQA(HFTask):

    def __init__(self, name, args):
        super().__init__(name, args)

    @property
    def metrics(self):
        return ['em', 'bleu']

    def _make_example(self, ex, **kwargs):
        example_id, question, used_queries, annotations, nq_answer = \
            ex['id'], ex['question'], ex['used_queries'], ex['annotations'], ex['nq_answer']
    
        # assert len(nq_answer) == 1, print(example_id, nq_answer)

        # TODO choosing only first answer for now
        answer = nq_answer[0]
    
        # process used_queries
        # assert len(used_queries['results']) == 1, print(example_id, used_queries)
        # TODO choosing only first used_query for now
        context = used_queries['query'][0] + ' <q> ' + ' <p> '.join(used_queries['results'][0]['snippet'])
        context_tokens = context.split(' ')[:180]
        context_tokens += ['<u>']
        context = ' '.join(context_tokens)
    
        if annotations['type'] == 'singleAnswer':
            # unambiguous question
            assert len(annotations['answer']) == 1
        else:
            # find the disambiguated question with the correct nq answer
            all_answers = annotations['qaPairs'][0]['answer']
            all_questions = annotations['qaPairs'][0]['question']
            assert len(all_answers) == len(all_questions)
            for q, a in zip(all_questions, all_answers):
                if a == nq_answer:
                    question = q
        
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)
    
    def get_splits(self, root, **kwargs):
        return HFDataset.return_splits(name=self.name, path=root, make_example=self._make_example, **kwargs)


@register_task('conll2003')
class CONLLNER(HFTask):
    num_labels = 9
    label_list = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    tagging_scheme = 'IOB2'
    
    def __init__(self, name, args):
        super().__init__(name, args)
    
    @property
    def metrics(self):
        return ['ner_f1', 'em', 'f1', 'pem']
    
    def _make_example(self, ex, **kwargs):
        example_id = ex['id']
        question = ' '.join(ex['tokens'])
        context = ''
        answer = ' '.join(map(lambda item: str(item), ex['ner_tags']))
        
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)
    
    def get_splits(self, root, **kwargs):
        return HFDataset.return_splits(name=self.name, path=root, make_example=self._make_example, **kwargs)

