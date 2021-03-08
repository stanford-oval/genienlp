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
import os

import torch
from datasets import load_dataset
from genienlp.tasks.generic_dataset import CQA, make_example_id

from .base_dataset import Split
from ..data_utils.example import Example

logger = logging.getLogger(__name__)


class AmbigQADataset(CQA):
    name = 'ambig_qa'
    
    def __init__(self, data, subsample=None, lower=False, **kwargs):
        
        skip_cache = kwargs.pop('kwargs', True)
        
        cache_name = os.path.join(os.path.dirname(data.cache_files[0]['filename']), data.split._name, str(subsample))
        examples = []
        
        if os.path.exists(cache_name) and not skip_cache:
            logger.info(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        for ex in data:
            example_id, question, used_queries, annotations, nq_answer = ex['id'], ex['question'], ex['used_queries'], ex['annotations'], ex['nq_answer']
            
            # assert len(nq_answer) == 1, print(example_id, nq_answer)
            
            #TODO choosing only first answer for now
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

            examples.append(Example.from_raw(make_example_id(self, len(examples)), context, question, answer, lower=lower))
            
            if subsample is not None and len(examples) >= subsample:
                break
        
        os.makedirs(os.path.dirname(cache_name), exist_ok=True)
        logger.info(f'Caching data to {cache_name}')
        torch.save(examples, cache_name)
    
        super().__init__(examples, **kwargs)

    
    @classmethod
    def return_splits(cls, root='.data', train='train', validation='validation', test='test', **kwargs):
        
        # download datasets and cache them
        train_data, validation_data, test_data = None, None, None
        train_path, validation_path, test_path = None, None, None
        if train:
            train_data = load_dataset(cls.name, split='train', cache_dir=root)
            train_path = train_data.cache_files[0]['filename']
        if validation:
            validation_data = load_dataset(cls.name, split='validation', cache_dir=root)
            validation_path = validation_data.cache_files[0]['filename']
        if test:
            test_data = load_dataset(cls.name, split='test', cache_dir=root)
            test_path = test_data.cache_files[0]['filename']

        train_data = None if train is None else cls(train_data, **kwargs)
        validation_data = None if validation is None else cls(validation_data, **kwargs)
        test_data = None if test is None else cls(test_data, **kwargs)

        return Split(train=train_data, eval=validation_data, test=test_data),\
               Split(train=train_path, eval=validation_path, test=test_path)
