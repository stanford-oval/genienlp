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

import os
import torch
import logging
from tqdm import tqdm

from ..base import BaseTask
from ..registry import register_task
from .. import generic_dataset
from ...data.example import Example

logger = logging.getLogger(__name__)


class AlmondDataset(generic_dataset.CQA):
    """Obtaining dataset for Almond semantic parsing task"""

    base_url = None

    def __init__(self, path, contextual=False, reverse_task=False, subsample=None, tokenize=None,
                 cached_path=None, skip_cache=False, **kwargs):
        cache_name = os.path.join(cached_path, os.path.basename(path), str(subsample))

        if os.path.exists(cache_name) and not skip_cache:
            logger.info(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            examples = []

            n = 0
            with open(path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    n += 1

            max_examples = min(n, subsample) if subsample is not None else n
            for i, line in tqdm(enumerate(open(path, 'r', encoding='utf-8')), total=max_examples):
                if i >= max_examples:
                    break

                parts = line.strip().split('\t')

                if contextual:
                    _id, context, sentence, target_code = parts
                else:
                    _id, sentence, target_code = parts
                    context = ''

                # remove BOM
                if sentence.startswith('\ufeff'):
                    sentence = sentence[1:]

                if contextual:
                    if reverse_task:
                        answer = sentence
                        question = target_code
                    else:
                        answer = target_code
                        question = sentence
                else:
                    # the question is irrelevant, so the question says English and ThingTalk even if we're doing
                    # a different language (like Chinese)
                    if reverse_task:
                        question = 'translate from thingtalk to english'
                        context = target_code
                        answer = sentence
                    else:
                        question = 'translate from english to thingtalk'
                        context = sentence
                        answer = target_code

                examples.append(Example.from_raw('almond/' + _id, context, question, answer,
                                                 tokenize=tokenize, lower=False))
                if len(examples) >= max_examples:
                    break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            logger.info(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        super().__init__(examples, **kwargs)

    @classmethod
    def splits(cls, root='.data', train='train', validation='eval', test='test', contextual=False, **kwargs):

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
        if contextual:
            name = 'contextual_almond'
        else:
            name = 'almond'
        path = os.path.join(root, name)

        aux_data = None
        if kwargs.get('curriculum', False):
            kwargs.pop('curriculum')
            aux_data = cls(os.path.join(path, 'aux' + '.tsv'), contextual=contextual, **kwargs)

        train_data = None if train is None else cls(
            os.path.join(path, train + '.tsv'), contextual=contextual, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation + '.tsv'), contextual=contextual, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test + '.tsv'), contextual=contextual, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data, aux_data)
                     if d is not None)

    @staticmethod
    def clean(path):
        pass


def is_entity(token):
    return token[0].isupper()


class BaseAlmondTask(BaseTask):
    """Base class for the Almond semantic parsing task
        i.e. natural language to formal language (ThingTalk) mapping"""

    def __init__(self, name, args):
        super().__init__(name, args)

    @property
    def metrics(self):
        return ['em', 'bleu']

    def _is_program_field(self, field_name):
        raise NotImplementedError()

    def tokenize(self, sentence, field_name=None):
        if not sentence:
            return [], []

        if self._is_program_field(field_name):
            mask = []
            in_string = False
            tokens = sentence.split(' ')
            for token in tokens:
                if token == '"':
                    in_string = not in_string
                    mask.append(False)
                else:
                    mask.append(in_string)

            assert len(tokens) == len(mask)
            return tokens, mask

        else:
            tokens = sentence.split(' ')
            mask = [not is_entity(token) for token in tokens]
            return tokens, mask

    def detokenize(self, tokenized, field_name=None):
        return ' '.join(tokenized)


@register_task('almond')
class Almond(BaseAlmondTask):
    """The Almond semantic parsing task
    i.e. natural language to formal language (ThingTalk) mapping"""

    def _is_program_field(self, field_name):
        return field_name == 'answer'

    def get_splits(self, root, **kwargs):
        return AlmondDataset.splits(root=root, tokenize=self.tokenize, **kwargs)


@register_task('contextual_almond')
class ContextualAlmond(BaseAlmondTask):
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def get_splits(self, root, **kwargs):
        return AlmondDataset.splits(root=root, tokenize=self.tokenize, contextual=True, **kwargs)


@register_task('reverse_almond')
class ReverseAlmond(BaseTask):
    """Reverse Almond semantic parsing task
    i.e. formal language to natural language mapping"""

    @property
    def metrics(self):
        return ['bleu', 'em', 'nf1']

    def _is_program_field(self, field_name):
        return field_name == 'context'

    def get_splits(self, root, **kwargs):
        return AlmondDataset.splits(root=root, reverse_task=True, tokenize=self.tokenize, **kwargs)
