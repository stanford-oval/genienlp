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
import os
import torch

from ..base import BaseTask
from ..registry import register_task
from .. import generic_dataset
from ...text.torchtext import data


logger = logging.getLogger(__name__)


class AlmondDataset(generic_dataset.CQA):
    """The Almond semantic parsing task"""

    base_url = None
    name = 'almond'

    def __init__(self, path, field, tokenize, reverse_task=False, subsample=None, **kwargs):
        fields = [(x, field) for x in self.fields]
        cached_path = kwargs.pop('cached_path')
        cache_name = os.path.join(cached_path, os.path.dirname(path).strip("/"), '.cache', os.path.basename(path),
                                  str(subsample))

        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        if reverse_task:
            question = 'Translate from ThingTalk to English'
        else:
            question = 'Translate from English to ThingTalk'

        skip_cache_bool = kwargs.pop('skip_cache_bool')
        if os.path.exists(cache_name) and not skip_cache_bool:
            logger.info(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            examples = []
            with open(path, 'r') as fp:
                for line in fp:
                    _id, sentence, target_code = line.strip().split('\t')
                    if reverse_task:
                        context = target_code
                        answer = sentence
                    else:
                        context = sentence
                        answer = target_code

                    context_question = generic_dataset.get_context_question(context, question)
                    examples.append(data.Example.fromlist(
                        [context, question, answer, generic_dataset.CONTEXT_SPECIAL, generic_dataset.QUESTION_SPECIAL, context_question], fields,
                        tokenize=tokenize))
                    if subsample is not None and len(examples) >= subsample:
                        break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            logger.info(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.context), len(ex.answer))

    @classmethod
    def splits(cls, fields, root='.data',
               train='train', validation='eval',
               test='test', tokenize=None, reverse_task=False, **kwargs):

        """Create dataset objects for splits of the ThingTalk dataset.
        Arguments:
            root: Root dataset storage directory. Default is '.data'.
            fields: A tuple containing the fields that will be used for data
                in each language.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'eval'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = os.path.join(root, cls.name)

        train_data = None if train is None else cls(
            os.path.join(path, train + '.tsv'), fields, tokenize=tokenize, reverse_task=reverse_task, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation + '.tsv'), fields, tokenize=tokenize, reverse_task=reverse_task, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test + '.tsv'), fields, tokenize=tokenize, reverse_task=reverse_task, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @staticmethod
    def clean(path):
        pass


@register_task('almond')
class Almond(BaseTask):
    def __init__(self, name, args):
        super().__init__(name, args)

        self._thingpedia = args.thingpedia
        self._grammar = None

    @property
    def metrics(self):
        return ['em', 'nem', 'nf1', 'fm', 'dm', 'bleu']

    def get_splits(self, field, root, **kwargs):
        return AlmondDataset.splits(
            fields=field, root=root, tokenize=self.tokenize, reverse_task=False, **kwargs)

    def tokenize(self, sentence):
        tokenized =  sentence.split(' ')

        if self._grammar is None:
            return tokenized

    def detokenize(self, tokenized):
        return ' '.join(tokenized)


@register_task('reverse_almond')
class ReverseAlmond(BaseTask):
    @property
    def metrics(self):
        return ['bleu', 'em', 'nem', 'nf1']

    def get_splits(self, field, root, **kwargs):
        return AlmondDataset.splits(
            fields=field, root=root, reverse_task=True, **kwargs)