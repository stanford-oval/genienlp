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
import json
from tqdm import tqdm
from collections import defaultdict

from ..base_task import BaseTask
from ..registry import register_task
from ..generic_dataset import CQA, context_answer_len, token_batch_fn, default_batch_fn
from ...data_utils.example import Example
from ...data_utils.database import Database, DOMAIN_TYPE_MAPPING

from ..base_dataset import Split

from wordfreq import zipf_frequency

logger = logging.getLogger(__name__)

quoted_pattern_with_space = re.compile(r'\"\s([^"]*?)\s\"')

ISO_to_LANG = {'en': 'English', 'en-US': 'English', 'fa': 'Persian', 'it': 'Italian', 'zh': 'Chinese',
               'hr': 'Croatian', 'ja': 'Japanese', 'ko': 'Korean', 'ru': 'Russian', 'es': 'Spanish',
               'sv': 'Swedish', 'tr': 'Turkish', 'hi': 'Hindi', 'fr': 'French', 'de': 'German',
               'pl': 'Polsih', 'ar': 'Arabic', 'vi': 'Vietnamese', 'ji': 'Yiddish', 'pt': 'Portuguese',
               'el': 'Greek', 'he': 'Hebrew', 'si': 'Sinhala', 'ta': 'Tamil', 'fi': 'Finnish', 'cs': 'Czech',
               'no': 'Norwegian', 'tl': 'Filipino', 'da': 'Danish'}

class AlmondDataset(CQA):
    """Obtaining dataset for Almond semantic parsing task"""

    base_url = None

    def __init__(self, path, *, make_example, subsample=None,
                 cached_path=None, skip_cache=False, cache_input_data=False,
                 split=None, **kwargs):
        
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
                examples.append(make_example(parts, dir_name, split, **kwargs))
                if len(examples) >= max_examples:
                    break
            
            if cache_input_data:
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
        
        train_data = None if train is None else cls(os.path.join(path, train + '.tsv'), split='train', **kwargs)
        validation_data = None if validation is None else cls(os.path.join(path, validation + '.tsv'), split='validation', **kwargs)
        test_data = None if test is None else cls(os.path.join(path, test + '.tsv'), split='test', **kwargs)

        aux_data = None
        do_curriculum = kwargs.get('curriculum', False)
        if do_curriculum:
            kwargs.pop('curriculum')
            aux_data = cls(os.path.join(path, 'aux' + '.tsv'), **kwargs)
        
        return Split(train=None if train is None else train_data,
                     eval=None if validation is None else validation_data,
                     test=None if test is None else test_data,
                     aux=None if do_curriculum is None else aux_data)
    

class BaseAlmondTask(BaseTask):
    """Base class for the Almond semantic parsing task
        i.e. natural language to formal language (ThingTalk) mapping"""

    def __init__(self, name, args):
        super().__init__(name, args)
        self.args = args
        self._preprocess_context = args.almond_preprocess_context
        
        # initialize the database
        if args.database and args.do_entity_linking:
            self._init_db()

    def is_entity(self, token):
        return token[0].isupper()

    def is_device(self, token):
        return token[0] == '@'

    def process_id(self, ex):
        id_ = ex.example_id.rsplit('/', 1)
        id_ = id_[0] if len(id_) == 1 else id_[1]
        # translated
        if id_[0] == 'T':
            id_ = id_[1:]
        return id_

    def _init_db(self):
        with open(self.args.database, 'r') as fin:
            db_data = json.load(fin)
            # lowercase all keys
            db_data_processed = {key.lower(): value for key, value in db_data.items()}
            self.db = Database(db_data_processed)
    
        self.TTtype2DBtype = dict()
        for domain in self.args.almond_domains:
            self.TTtype2DBtype.update(DOMAIN_TYPE_MAPPING[domain])
    
    @property
    def metrics(self):
        return ['em', 'bleu']

    def _is_program_field(self, field_name):
        raise NotImplementedError()
    
    def is_contextual(self):
        raise NotImplementedError()

    def _make_example(self, parts, dir_name, **kwargs):
        raise NotImplementedError()

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, **kwargs)
    
    def preprocess_context(self, sentence):
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
        return preprocessed_context
    
    def collect_answer_entity_types(self, answer):
        entity2type = dict()
        answer_entities = quoted_pattern_with_space.findall(answer)
        for ent in answer_entities:
            # this is thingtalk specific and also domain specific
            # (music with syntax: ... param:inAlbum:Entity(org.schema.Music:MusicAlbum) == " XXXX " ... )
            # (spotify with syntax: ... param:artists contains " XXX " ^^com.spotify:artist and param:id =~ " XXX " ...)
            # this needs to be changed if annotations convention changes
        
            # assume first syntax
            idx = answer.index('" ' + ent + ' "')
            schema_entity_type = answer[:idx].split()[-2]
        
            if schema_entity_type.startswith('param:'):
                schema_entity_type = schema_entity_type.strip('()').rsplit(':', 1)[1]
            else:
                # check for ^^ syntax
                schema_entity_type = answer[idx + len('" ' + ent + ' "'):].split()
                if len(schema_entity_type) == 1:
                    schema_entity_type = 'id'
                else:
                    schema_entity_type = schema_entity_type[0]
                if schema_entity_type.startswith('^^'):
                    schema_entity_type = schema_entity_type.rsplit(':', 1)[1]
                else:
                    schema_entity_type = self.db.unk_type
        
            if schema_entity_type not in self.TTtype2DBtype.keys():
                schema_type = self.db.unk_type
            else:
                schema_type = self.TTtype2DBtype[schema_entity_type]
        
            entity2type[ent] = schema_type
        
        return entity2type

    def find_types(self, tokens, split, answer, no_oracle=False):
        # we only need to do lookup for test split as entity types can be retrieved for train and eval sets from the program
        # this will speed up the process significantly
        if no_oracle or split in ['test']:
            tokens_type_ids = self.db.lookup(tokens)
        else:
            if self.args.retrieve_method == 'database':
                tokens_type_ids = self.db.lookup(tokens)
            else:
                entity2type = self.collect_answer_entity_types(answer)
                tokens_type_ids = self.db.lookup(tokens, subset=entity2type, retrieve_method=self.args.retrieve_method)
    
        return tokens_type_ids
    
    def find_freqs(self, tokens, tokens_type_ids):
        token_freqs = []
        for token, token_type_id in zip(tokens, tokens_type_ids):
            if token_type_id == self.db.type2id[self.db.unk_type]:
                token_freqs.append(1.0)
            else:
                token_freqs.append(1.0 / (zipf_frequency(token, 'en') + 1e-3))
        return token_freqs
        
    def tokenize(self, sentence, split=None, field_name=None, answer=None, no_oracle=False):

        if not sentence:
            return [], [], []

        if self.force_subword_tokenize:
            tokens = sentence.split(' ')
        else:
            tokens = [t for t in sentence.split(' ') if len(t) > 0]
            if self._preprocess_context and field_name in ('context', 'context_question'):
                tokens = self.preprocess_context(sentence)
                
        tokens_type_ids = []
        token_freqs = []
        if self.args.do_entity_linking and field_name in ('question', 'context', 'context_question'):
            if 'type' in self.args.features:
                tokens_type_ids = self.find_types(tokens, split, answer, no_oracle)
            if 'freq' in self.args.features:
                token_freqs = self.find_freqs(tokens, tokens_type_ids)
        
        if self.args.verbose and self.args.do_entity_linking and \
                ((self.is_contextual() and field_name == 'question') or (not self.is_contextual() and field_name == 'context')) and \
                split == 'validation':
            print()
            print(*[f'entity: {token}\ttype: {token_type}' for token, token_type in zip(tokens, tokens_type_ids)], sep='\n')

        zip_list = []
        if tokens_type_ids:
            assert len(tokens) == len(tokens_type_ids)
            zip_list.append(tokens_type_ids)
        if token_freqs:
            assert len(tokens) == len(token_freqs)
            zip_list.append(token_freqs)

        if self.force_subword_tokenize:
            return tokens, None, list(zip(*zip_list))

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

            return tokens, mask, list(zip(*zip_list))

        else:
            mask = [not self.is_entity(token) and not self.is_device(token) for token in tokens]
            return tokens, mask, list(zip(*zip_list))

    def detokenize(self, tokenized, field_name=None):
        return ' '.join(tokenized)


@register_task('almond')
class Almond(BaseAlmondTask):
    """The Almond semantic parsing task
    i.e. natural language to formal language (ThingTalk) mapping"""

    def _is_program_field(self, field_name):
        return field_name == 'answer'

    def is_contextual(self):
        return False

    def _make_example(self, parts, dir_name=None, split=None, no_oracle=False, **kwargs):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        _id, sentence, target_code = parts
        question = 'translate from english to thingtalk'
        context = sentence
        answer = target_code

        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, split=split, no_oracle=no_oracle, lower=False)


@register_task('contextual_almond')
class ContextualAlmond(BaseAlmondTask):
    """Contextual Almond semantic parsing task
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def is_contextual(self):
        return True
    
    def _make_example(self, parts, dir_name=None, split=None, no_oracle=False, **kwargs):
        _id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, split=split, no_oracle=no_oracle, lower=False)


@register_task('reverse_almond')
class ReverseAlmond(BaseTask):
    """Reverse Almond semantic parsing task
    i.e. formal language to natural language mapping"""

    @property
    def metrics(self):
        return ['bleu', 'em']

    def is_contextual(self):
        return False
    
    def _is_program_field(self, field_name):
        return field_name == 'context'

    def _make_example(self, parts, dir_name=None, split=None, no_oracle=False, **kwargs):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        _id, sentence, target_code = parts
        question = 'translate from thingtalk to english'
        context = target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, split=None, no_oracle=no_oracle, lower=False)


@register_task('almond_dialogue_nlu')
class AlmondDialogueNLU(BaseAlmondTask):
    """Multi-turn NLU task for Almond dialogues
    (translate the user utterance to a formal representation, given the current
    state of the conversation)
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def is_contextual(self):
        return True
    
    def _make_example(self, parts, dir_name=None, split=None, no_oracle=False, **kwargs):
        _id, context, sentence, target_code = parts

        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, split=split, no_oracle=no_oracle, lower=False)

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

    def is_contextual(self):
        return True
    
    def _make_example(self, parts, dir_name=None, split=None, no_oracle=False, **kwargs):
        _id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, split=split, no_oracle=no_oracle, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)


@register_task('almond_dialogue_nlg')
class AlmondDialogueNLG(BaseAlmondTask):
    """Multi-turn NLG task for Almond dialogues
    (generate the system utterance, given the current state of the conversation
    and the desired system dialogue act)
    """
    def _is_program_field(self, field_name):
        return field_name == 'context'

    def is_contextual(self):
        return True
    
    @property
    def metrics(self):
        return ['bleu']

    def _make_example(self, parts, dir_name=None, split=None, no_oracle=False, **kwargs):
        # the question is irrelevant for this task
        _id, context, sentence, target_code = parts
        question = 'what should the agent say ?'
        context = context + ' ' + target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, split=split, no_oracle=no_oracle, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)


@register_task('almond_dialogue_policy')
class AlmondDialoguePolicy(BaseAlmondTask):
    """Multi-turn dialogue policy task for Almond dialogues
    (generate the next dialogue act, given the current state of the conversation)
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def is_contextual(self):
        return True
    
    @property
    def metrics(self):
        return ['em', 'bleu']

    def _make_example(self, parts, dir_name=None, split=None, no_oracle=False, **kwargs):
        # the question is irrelevant for this task, and the sentence is intentionally ignored
        _id, context, _sentence, target_code = parts
        question = 'what should the agent do ?'
        context = context
        answer = target_code
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, split=split, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)
    

class BaseAlmondMultiLingualTask(BaseAlmondTask):
    """ Base task for MultiLingual Almond
    """
    def get_train_processed_ids(self, split):
        all_ids = []
        for ex in split.examples:
            all_ids.append(self.process_id(ex))
        return all_ids

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
            
            sort_key_fn = self.process_id
            batch_size_fn = default_batch_fn
        else:
            sort_key_fn = context_answer_len
            batch_size_fn = token_batch_fn
            
        groups = len(all_datasets) if kwargs.get('sentence_batching') else None
        
        if kwargs.get('separate_eval') and (all_datasets[0].eval or all_datasets[0].test):
            return all_datasets
        else:
            return self.combine_datasets(all_datasets, sort_key_fn, batch_size_fn, used_fields, groups)


@register_task('almond_multilingual')
class AlmondMultiLingual(BaseAlmondMultiLingualTask):
    
    def _is_program_field(self, field_name):
        return field_name == 'answer'

    def is_contextual(self):
        return False

    @property
    def metrics(self):
        return ['em', 'bleu']
    
    def _make_example(self, parts, dir_name, split=None, no_oracle=False, **kwargs):
        _id, sentence, target_code = parts
        language = ISO_to_LANG.get(dir_name, 'English').lower()
        if kwargs.get('lang_as_question'):
            question = 'translate from {} to thingtalk'.format(language)
        else:
            question = 'translate from english to thingtalk'
        context = sentence
        answer = target_code
        return Example.from_raw(self.name + '/' + dir_name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, split=split, no_oracle=no_oracle, lower=False)


@register_task('almond_dialog_multilingual_nlu')
class AlmondDialogMultiLingualNLU(BaseAlmondMultiLingualTask):
    """Multi-turn NLU task (user and agent) for MultiLingual Almond dialogues
    """

    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')
    
    def is_contextual(self):
        return True
    
    @property
    def metrics(self):
        return ['em', 'bleu']

    def _make_example(self, parts, dir_name=None, split=None, no_oracle=False, **kwargs):
        _id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + dir_name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, split=split, no_oracle=no_oracle, lower=False)


@register_task('almond_dialog_multilingual_nlg')
class AlmondDialogMultiLingualNLG(BaseAlmondTask):
    """Multi-turn NLG task (agent) for MultiLingual Almond dialogues
    """
    def _is_program_field(self, field_name):
        return field_name == 'context'
    
    def is_contextual(self):
        return True

    @property
    def metrics(self):
        return ['bleu']

    def _make_example(self, parts, dir_name=None, split=None, no_oracle=False, **kwargs):
        # the question is irrelevant for this task
        _id, context, sentence, target_code = parts
        question = 'what should the agent say ?'
        context = context + ' ' + target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + dir_name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, split=split, no_oracle=no_oracle, lower=False)

