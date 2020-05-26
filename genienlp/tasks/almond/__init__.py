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
import re
from tqdm import tqdm
from collections import defaultdict

from ..base_task import BaseTask
from ..registry import register_task
from ..generic_dataset import CQA, context_answer_len, token_batch_fn, default_batch_fn
from ...data_utils.example import Example

from ..base_dataset import Split

logger = logging.getLogger(__name__)

ISO_to_LANG = {'en': 'English', 'en-US': 'English', 'fa': 'Farsi', 'it': 'Italian', 'zh': 'Chinese'}

class AlmondDataset(CQA):
    """Obtaining dataset for Almond semantic parsing task"""

    base_url = None

    def __init__(self, path, *, make_example, subsample=None, cached_path=None, skip_cache=False, **kwargs):
        
        #TODO fix cache_path for multilingual task
        cache_name = os.path.join(cached_path, os.path.basename(path), str(subsample))
        dir_name = os.path.basename(os.path.dirname(path))

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
                parts = line.strip().split('\t')
                examples.append(make_example(parts, dir_name))
                if len(examples) >= max_examples:
                    break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            logger.info(f'Caching data to {cache_name}')
            torch.save(examples, cache_name)

        super().__init__(examples, **kwargs)
        

    @classmethod
    def return_splits(cls, path, train='train', validation='eval', test='test', **kwargs):

        """Create dataset objects for splits of the ThingTalk dataset.
        Arguments:
            path: path to directory where data splits reside
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'eval'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        
        train_data = None if train is None else cls(os.path.join(path, train + '.tsv'), **kwargs)
        validation_data = None if validation is None else cls(os.path.join(path, validation + '.tsv'), **kwargs)
        test_data = None if test is None else cls(os.path.join(path, test + '.tsv'), **kwargs)

        aux_data = None
        do_curriculum = kwargs.get('curriculum', False)
        if do_curriculum:
            kwargs.pop('curriculum')
            aux_data = cls(os.path.join(path, 'aux' + '.tsv'), **kwargs)
        
        return Split(train=None if train is None else train_data,
                     eval=None if validation is None else validation_data,
                     test=None if test is None else test_data,
                     aux=None if do_curriculum is None else aux_data)
    

def is_entity(token):
    return token[0].isupper()

def is_device(token):
    return token[0] == '@'

def process_id(ex):
    id_ = ex.example_id.rsplit('/', 1)
    id_ = id_[0] if len(id_) == 1 else id_[1]
    # translated
    if id_[0] == 'T':
        id_ = id_[1:]
    return id_


class BaseAlmondTask(BaseTask):
    """Base class for the Almond semantic parsing task
        i.e. natural language to formal language (ThingTalk) mapping"""

    def __init__(self, name, args):
        super().__init__(name, args)
        self._preprocess_context = args.almond_preprocess_context

    @property
    def metrics(self):
        return ['em', 'bleu']

    def _is_program_field(self, field_name):
        raise NotImplementedError()

    def _make_example(self, parts, dir_name=None):
        raise NotImplementedError()

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, **kwargs)

    def tokenize(self, sentence, field_name=None):
        if not sentence:
            return [], []

        if self.force_subword_tokenize:
            return sentence.split(' '), None

        tokens = sentence.split(' ')
        if self._preprocess_context and field_name in ('context', 'context_question'):
            preprocessed_context = []
            for token in sentence.split(' '):
                if token.startswith('@'):
                    word = '_'.join(token.rsplit('.', maxsplit=2)[1:3]).lower()
                    preprocessed_context += word.split('_')
                elif token.startswith('param:'):
                    word = token[len('param:'):]
                    preprocessed_context += word.split('_')
                elif token.startswith('enum:'):
                    word = token[len('enum:'):]
                    preprocessed_context += word.split('_')
                else:
                    preprocessed_context.append(token)
            tokens = preprocessed_context

        if self._is_program_field(field_name):
            mask = []
            in_string = False
            for token in tokens:
                if token == '"':
                    in_string = not in_string
                    mask.append(False)
                else:
                    mask.append(in_string)

            assert len(tokens) == len(mask)
            return tokens, mask

        else:
            mask = [not is_entity(token) and not is_device(token) for token in tokens]
            return tokens, mask

    def detokenize(self, tokenized, field_name=None):
        return ' '.join(tokenized)


@register_task('almond')
class Almond(BaseAlmondTask):
    """The Almond semantic parsing task
    i.e. natural language to formal language (ThingTalk) mapping"""

    def _is_program_field(self, field_name):
        return field_name == 'answer'

    def _make_example(self, parts, dir_name=None):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        _id, sentence, target_code = parts
        question = 'translate from english to thingtalk'
        context = sentence
        answer = target_code
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)


@register_task('contextual_almond')
class ContextualAlmond(BaseAlmondTask):
    """Contextual Almond semantic parsing task
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def _make_example(self, parts, dir_name=None):
        _id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)


@register_task('reverse_almond')
class ReverseAlmond(BaseTask):
    """Reverse Almond semantic parsing task
    i.e. formal language to natural language mapping"""

    @property
    def metrics(self):
        return ['bleu', 'em']

    def _is_program_field(self, field_name):
        return field_name == 'context'

    def _make_example(self, parts, dir_name=None):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        _id, sentence, target_code = parts
        question = 'translate from thingtalk to english'
        context = target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)


@register_task('almond_dialogue_nlu')
class AlmondDialogueNLU(BaseAlmondTask):
    """Multi-turn NLU task for Almond dialogues
    (translate the user utterance to a formal representation, given the current
    state of the conversation)
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def _make_example(self, parts, dir_name=None):
        _id, context, sentence, target_code = parts

        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/user'), make_example=self._make_example, **kwargs)


@register_task('almond_dialogue_nlu_agent')
class AlmondDialogueNLUAgent(BaseAlmondTask):
    """Multi-turn NLU task for Almond dialogues, for the agent utterance
    (translate the agent utterance to a formal representation, given the current
    state of the conversation).
    This is used to facilitate annotation of human-human dialogues.
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def _make_example(self, parts, dir_name=None):
        _id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)


@register_task('almond_dialogue_nlg')
class AlmondDialogueNLG(BaseAlmondTask):
    """Multi-turn NLG task for Almond dialogues
    (generate the system utterance, given the current state of the conversation
    and the desider system dialogue act)
    """
    def _is_program_field(self, field_name):
        return field_name == 'context'

    @property
    def metrics(self):
        return ['bleu']

    def _make_example(self, parts, dir_name=None):
        # the question is irrelevant for this task
        _id, context, sentence, target_code = parts
        question = 'what should the agent say ?'
        context = context + ' ' + target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)


@register_task('almond_dialogue_policy')
class AlmondDialoguePolicy(BaseAlmondTask):
    """Multi-turn dialogue policy task for Almond dialogues
    (generate the next dialogue act, given the current state of the conversation)
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    @property
    def metrics(self):
        return ['em', 'bleu']

    def _make_example(self, parts, dir_name=None):
        # the question is irrelevant for this task, and the sentence is intentionally ignored
        _id, context, _sentence, target_code = parts
        question = 'what should the agent do ?'
        context = context
        answer = target_code
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)
    

@register_task('almond_multilingual')
class AlmondMultiLingual(BaseAlmondTask):
    """Multi-Language task for Almond
    """

    def _is_program_field(self, field_name):
        return field_name == 'answer'

    @property
    def metrics(self):
        return ['em', 'bleu']

    def _make_example(self, parts, dir_name):
        _id, sentence, target_code = parts
        language = ISO_to_LANG.get(dir_name, 'English')
        question = 'translate from {} to thingtalk'.format(language)
        context = sentence
        answer = target_code
        return Example.from_raw(self.name + '/' + dir_name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)
    
    def get_train_processed_ids(self, split):
        all_ids = []
        for ex in split.examples:
            all_ids.append(process_id(ex))
        return all_ids
        
    def get_splits(self, root, **kwargs):
        all_datasets = []
        # number of directories to read data from
        all_dirs = kwargs['all_dirs'].split('+')
        
        for dir in all_dirs:
            almond_dataset = AlmondDataset.return_splits(path=os.path.join(root, 'almond/multilingual/{}'.format(dir)),
                                                         make_example=self._make_example, **kwargs)
            all_datasets.append(almond_dataset)
            
        used_fields = [field for field in all_datasets[0]._fields if getattr(all_datasets[0], field) is not None]
        
        assert len(all_datasets) >= 1
        if kwargs.get('sentence_batching'):
            for field in used_fields:
                lengths = list(map(lambda dataset: len(getattr(dataset, field)), all_datasets))
                assert len(set(lengths)) == 1, 'When using sentence batching your datasets should have the same size.'
            if 'train' in used_fields:
                ids_sets = list(map(lambda dataset: set(self.get_train_processed_ids(dataset.train)), all_datasets))
                id_set_base = set(ids_sets[0])
                for id_set in ids_sets:
                    assert set(id_set) == id_set_base, 'When using sentence batching your datasets should have matching ids'
            
            sort_key_fn = process_id
            batch_size_fn = default_batch_fn
        else:
            sort_key_fn = context_answer_len
            batch_size_fn = token_batch_fn
            
        groups = len(all_datasets) if kwargs.get('sentence_batching') else None
        
        if kwargs.get('separate_eval') and (all_datasets[0].eval or all_datasets[0].test):
            return all_datasets
        else:
            return self.combine_datasets(all_datasets, sort_key_fn, batch_size_fn, used_fields, groups)

    def combine_datasets(self, datasets, sort_key_fn, batch_size_fn, used_fields, groups):
        splits = defaultdict()
        
        for field in used_fields:
            all_examples = []
            for dataset in datasets:
                all_examples.extend(getattr(dataset, field).examples)
                
            splits[field] = CQA(all_examples, sort_key_fn=sort_key_fn, batch_size_fn=batch_size_fn, groups=groups)
        
        return Split(train=splits.get('train'),
                     eval=splits.get('eval'),
                     test=splits.get('test'),
                     aux=splits.get('aux'))
