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
import sys
from tqdm import tqdm

from ..base import BaseTask
from ..registry import register_task
from .. import generic_dataset
from ...text.torchtext import data
from ...utils.lang_utils import *


from .grammar import thingtalk, plainthingtalk, posthingtalk

logger = logging.getLogger(__name__)


class AlmondDataset(generic_dataset.CQA):
    """Obtaining dataset for Almond semantic parsing task"""

    base_url = None

    def __init__(self, path, field, tokenize, contextual=False, thingpedia_in_context=False, reverse_task=False, subsample=None, **kwargs):

        fields = [(x, field) for x in self.fields]
        cached_path = kwargs.pop('cached_path')
        cache_name = os.path.join(cached_path, os.path.dirname(path).strip("/"), '.cache', os.path.basename(path), str(subsample))

        skip_cache_bool = kwargs.pop('skip_cache_bool')
        if os.path.exists(cache_name) and not skip_cache_bool:
            logger.info(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            examples = []

            n = 0
            with open(path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    n += 1


            if thingpedia_in_context:
                thingpedia = kwargs.pop('thingpedia')
                words_list = extract_words(thingpedia)
            else:
                words_list = None

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

                if thingpedia_in_context:
                    if contextual:
                        context = ' '.join(words_list) + ' || ' + context
                    else:
                        context = ' '.join(words_list)

                if contextual or thingpedia_in_context:
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
                        question = target_question if target_question is not None else 'Translate from ThingTalk to English'
                        context = target_code
                        answer = sentence
                    else:
                        question = target_question if target_question is not None else 'Translate from English to ThingTalk'
                        context = sentence
                        answer = target_code


                context_question = generic_dataset.get_context_question(context, question)
                examples.append(data.Example.fromlist(
                    [context, question, answer, generic_dataset.CONTEXT_SPECIAL, generic_dataset.QUESTION_SPECIAL, context_question], fields,
                    tokenize=tokenize))
                if len(examples) >= max_examples:
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
               test='test', contextual=False, **kwargs):

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
            aux_data = cls(os.path.join(path, 'aux' + '.tsv'), fields, contextual=contextual, **kwargs)

        train_data = None if train is None else cls(
            os.path.join(path, train + '.tsv'), fields, contextual=contextual, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation + '.tsv'), fields, contextual=contextual, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test + '.tsv'), fields, contextual=contextual, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data, aux_data)
                     if d is not None)

    @staticmethod
    def clean(path):
        pass


class BaseAlmondTask(BaseTask):
    """Base class for the Almond semantic parsing task
        i.e. natural language to formal language (ThingTalk) mapping"""

    def __init__(self, name, args):
        super().__init__(name, args)

        self._thingpedia = args.thingpedia
        self._grammar = None
        self._grammar_direction = None
        self.args = args

        if args.almond_grammar:
            self._grammar_direction = args.almond_grammar.split('.')[-1]
            if args.almond_grammar.startswith('typeless.'):
                self._grammar = plainthingtalk.PlainThingTalkGrammar(self._thingpedia, grammar_include_types=False,
                                                                     logger=logger)
            elif args.almond_grammar.startswith('plain.'):
                self._grammar = plainthingtalk.PlainThingTalkGrammar(self._thingpedia, grammar_include_types=True,
                                                                     logger=logger)
            elif args.almond_grammar.startswith('pos.typeless.'):
                self._grammar = posthingtalk.PosThingTalkGrammar(self._thingpedia, grammar_include_types=False,
                                                                 logger=logger)
            elif args.almond_grammar.startswith('pos.'):
                self._grammar = posthingtalk.PosThingTalkGrammar(self._thingpedia, grammar_include_types=True,
                                                                 logger=logger)
            else:
                self._grammar = thingtalk.ThingTalkGrammar(self._thingpedia, logger=logger)

    @property
    def metrics(self):
        return ['em', 'nem', 'nf1', 'fm', 'dm', 'bleu']


    def get_splits(self, field, root, **kwargs):
        if self.args.question is not None:
            kwargs['question'] = self.args.question
        return AlmondDataset.splits(
            fields=field, root=root, tokenize=self.tokenize, reverse_task=False, **kwargs)

    def tokenize(self, sentence, field_name=None):
        if not sentence:
            return []
        tokenized = sentence.split(' ')

        if self._grammar is None or field_name != 'answer':
            return tokenized
        else:
            return self._grammar.preprocess_program(tokenized, direction=self._grammar_direction)

    def detokenize(self, tokenized, field_name=None):
        if self._grammar is None or field_name != 'answer':
            return ' '.join(tokenized)
        else:
            return ' '.join(self._grammar.reconstruct_program(tokenized, direction=self._grammar_direction, ignore_errors=True))


@register_task('almond')
class Almond(BaseAlmondTask):
    """The Almond semantic parsing task
    i.e. natural language to formal language (ThingTalk) mapping"""

    def get_splits(self, field, root, **kwargs):
        return AlmondDataset.splits(
            fields=field, root=root, tokenize=self.tokenize, **kwargs)


@register_task('almond_with_thingpedia_as_context')
class AlmondWithThingpediaAsContext(BaseAlmondTask):

    def __init__(self, name, args):
        super().__init__(name, args)
        self._default_context = ' '.join(extract_words(self._thingpedia))

    @property
    def default_context(self):
        return self._default_context

    def get_splits(self, field, root, **kwargs):
        return AlmondDataset.splits(
            fields=field, root=root, tokenize=self.tokenize, thingpedia_in_context=True, thingpedia=self._thingpedia, **kwargs)


@register_task('contextual_almond')
class ContextualAlmond(BaseAlmondTask):

    def get_splits(self, field, root, **kwargs):
        return AlmondDataset.splits(
            fields=field, root=root, tokenize=self.tokenize, contextual=True, **kwargs)


@register_task('reverse_almond')
class ReverseAlmond(BaseTask):
    """Reverse Almond semantic parsing task
    i.e. formal language to natural language mapping"""

    @property
    def metrics(self):
        return ['bleu', 'em', 'nem', 'nf1']

    def get_splits(self, field, root, **kwargs):
        return AlmondDataset.splits(
            fields=field, root=root, reverse_task=True, **kwargs)
