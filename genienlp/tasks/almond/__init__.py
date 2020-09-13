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
import math
from tqdm import tqdm
from collections import defaultdict
# import multiprocessing as mp
import pathos.multiprocessing as mp


from ..base_task import BaseTask
from ..registry import register_task
from ..generic_dataset import CQA, context_answer_len, token_batch_fn, default_batch_fn
from ...data_utils.example import Example
from ...data_utils.database import Database, LocalElasticDatabase, RemoteElasticDatabase, DOMAIN_TYPE_MAPPING
from ...data_utils.bootleg import BootlegAnnotator
from ...util import es_dump_type2id, is_chinese_char

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


def process(args):
    path = args['in_file']
    chunk_size = args['chunk_size']
    dir_name = args['dir_name']
    example_batch_size = args['example_batch_size']
    make_process_example = args['make_process_example']
    kwargs = args['kwargs']
    
    chunk_examples = []
    
    batch = []
    last_batch = False
    for i, line in tqdm(enumerate(open(path, 'r', encoding='utf-8')), total=chunk_size):
        parts = line.strip().split('\t')
        batch.append(parts)
        if len(chunk_examples) + example_batch_size > chunk_size:
            # trim batch
            batch = batch[:chunk_size - len(chunk_examples)]
            last_batch = True
        if len(batch) % example_batch_size != 0 and not last_batch:
            continue
        chunk_examples.extend(make_process_example(batch, dir_name, **kwargs))
        batch = []
    
    return chunk_examples

def chunk_file(input_src, chunk_files, chunk_size, num_chunks):
    chunk_id = 0
    num_lines_in_chunk = 0
    all_out_files = [open(chunk_files[chunk_id], 'w') for chunk_id in range(num_chunks)]
    with open(input_src, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            all_out_files[chunk_id].write(line)
            num_lines_in_chunk += 1
            if num_lines_in_chunk == chunk_size:
                chunk_id += 1
                num_lines_in_chunk = 0
                if chunk_id == num_chunks:
                    break

    for file in all_out_files:
        file.close()


class AlmondDataset(CQA):
    """Obtaining dataset for Almond semantic parsing task"""

    base_url = None

    def __init__(self, path, *, make_example, subsample=None,
                 cached_path=None, skip_cache=False, cache_input_data=False, **kwargs):
    
        #TODO fix cache_path for multilingual task
        cache_name = os.path.join(cached_path, os.path.basename(path), str(subsample))
        dir_name = os.path.basename(os.path.dirname(path))
        
        example_batch_size = kwargs.get('example_batch_size', 1)
        num_processes = kwargs.get('num_workers', int(mp.cpu_count()))

        if os.path.exists(cache_name) and not skip_cache:
            logger.info(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            n = 0
            with open(path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    n += 1

            max_examples = min(n, subsample) if subsample is not None else n

            logger.info(f'Using {num_processes} workers...')
            chunk_size = int(math.ceil(max_examples / num_processes))
            num_chunks = int(math.ceil(max_examples / chunk_size))
            
            base_path, extension = path.rsplit('.', 1)
            
            chunk_file_paths = [f'{base_path}_{chunk_id}.tsv' for chunk_id in range(num_chunks)]
            chunk_file(path, chunk_file_paths, chunk_size, num_chunks)
            num_processes = min(num_processes, num_chunks)
            
            with mp.Pool(processes=num_processes) as pool:
                process_args = [{'in_file': chunk_file_paths[i], 'chunk_size': chunk_size, 'dir_name': dir_name,
                                'example_batch_size': example_batch_size, 'make_process_example': make_example,
                                'kwargs': kwargs} for i in range(num_chunks)]
                results = pool.map(process, process_args)
            
            # merge all results
            examples = [item for sublist in results for item in sublist]
            
            for file in chunk_file_paths:
                os.remove(file)

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
    

class BaseAlmondTask(BaseTask):
    """Base class for the Almond semantic parsing task
        i.e. natural language to formal language (ThingTalk) mapping"""

    def __init__(self, name, args):
        super().__init__(name, args)
        self.args = args
        self._preprocess_context = args.almond_preprocess_context
        
        # initialize the database
        if args.do_entity_linking:
            self._init_db()
            
            if self.args.retrieve_method == 'bootleg':
                self._init_bootleg()

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
    
    def _init_bootleg(self):
        # bootleg database is stored on disk in json format
        # TODO: integrate with ES
        if self.args.bootleg_device is not None:
            bootleg_device = self.args.bootleg_device
        else:
            if not torch.cuda.is_available() or len(self.args.devices) == 0:
                bootleg_device = 'cpu'
            else:
                bootleg_device = 'cuda'
        from bootleg.utils.parser_utils import get_full_config
        bootleg_dir = self.args.bootleg_input_dir
        config_path = f'{bootleg_dir}/bootleg_wiki/bootleg_config.json'
        config_args = get_full_config(config_path)
        self.bootleg_annot = BootlegAnnotator(config_args, device=bootleg_device, bootleg_dir=bootleg_dir)
        self.bootleg_annot.bootleg_es = self.db

    def _init_db(self):
        if self.args.database_type in ['json', 'local-elastic']:
            with open(self.args.database, 'r') as fin:
                db_data = json.load(fin)
                # lowercase all keys
                db_data_processed = {key.lower(): value for key, value in db_data.items()}
        elif self.args.database_type == 'remote-elastic':
            with open(self.args.elastic_config, 'r') as fin:
                es_config = json.load(fin)
            type2id = dict()
            if self.args.type2id_dict:
                with open(self.args.type2id_dict, 'r') as fin:
                    type2id = json.load(fin)
    
        if self.args.database_type == 'json':
            self.db = Database(db_data_processed)
        elif self.args.database_type == 'local-elastic':
            self.db = LocalElasticDatabase(db_data_processed)
        elif self.args.database_type == 'remote-elastic':
            self.db = RemoteElasticDatabase(es_config, type2id)
            if self.args.create_type_mapping:
                es_dump_type2id(self.db)
    
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
            if token.startswith('@') and '.' in token:
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

    def batch_find_types(self, tokens_list, answer_list):
    
        all_tokens_type_ids = []
        if self.args.retrieve_method == 'bootleg':
            # for i, token in enumerate(tokens_list):
            #     all_tokens_type_ids.append(self.bootleg_annot.return_type_ids(token))
            all_tokens_type_ids = self.bootleg_annot.batch_return_type_ids(tokens_list)
        else:
            if self.args.database_type == 'json':
                if self.args.retrieve_method == 'lookup':
                    all_tokens_type_ids = self.db.batch_lookup(tokens_list, lookup_method=self.args.lookup_method)
                elif self.args.retrieve_method == 'oracle':
                    entity2type = self.collect_answer_entity_types(answer_list)
                    all_tokens_type_ids = self.db.batch_lookup(tokens_list, subset=entity2type,
                                                     retrieve_method='oracle', lookup_method=self.args.lookup_method)
            else:
                all_tokens_type_ids = self.db.batch_lookup(tokens_list, allow_fuzzy=self.args.allow_fuzzy)
        
        return all_tokens_type_ids
    
    def find_freqs(self, tokens, tokens_type_ids):
        token_freqs = []
        for token, token_type_id in zip(tokens, tokens_type_ids):
            if token_type_id == self.db.type2id[self.db.unk_type]:
                token_freqs.append(1.0)
            else:
                token_freqs.append(1.0 / (zipf_frequency(token, 'en') + 1e-3))
        return token_freqs
    
    def _detokenize_cjk_chars(self, sentence):
        output = []
        i = 0
        while i < len(sentence):
            output.append(sentence[i])
            # skip space after cjk chars only if followed by another cjk char
            if is_chinese_char(ord(sentence[i])) \
                and i+1 < len(sentence) and sentence[i+1] == ' ' \
                and i+2 < len(sentence) and is_chinese_char(ord(sentence[i+2])):
                    i += 2
            else:
                i += 1
        return "".join(output)
    
    def tokenize(self, sentence_list, field_name=None, answer_list=None):
        all_tokens, all_masks, all_features = [], [], []
        for sentence, answer in zip(sentence_list, answer_list):
            tokens, masks = self.tokenize_single(sentence, field_name)
            all_tokens.append(tokens)
            all_masks.append(masks)

        all_tokens_type_ids = []
        all_token_freqs = []
        if self.args.do_entity_linking and field_name in ('question', 'context', 'context_question'):
            if 'type' in self.args.features:
                all_tokens_type_ids = self.batch_find_types(all_tokens, answer_list)
            if 'freq' in self.args.features:
                for tokens, tokens_type_ids in zip(all_tokens, all_tokens_type_ids):
                    all_token_freqs.append(self.find_freqs(tokens, tokens_type_ids))
        if self.args.verbose and self.args.do_entity_linking and \
                ((self.is_contextual() and field_name == 'question') or
                 (not self.is_contextual() and field_name == 'context')):
            for tokens, tokens_type_ids in zip(all_tokens, all_tokens_type_ids):
                print()
                print(*[f'entity: {token}\ttype: {token_type}' for token, token_type in zip(tokens, tokens_type_ids)],
                      sep='\n')
        
        all_features = []
        for i in range(len(sentence_list)):
            zip_list = []
            tokens = all_tokens[i]
            if all_tokens_type_ids:
                tokens_type_ids = all_tokens_type_ids[i]
                assert len(tokens) == len(tokens_type_ids)
                zip_list.append(tokens_type_ids)
            if all_token_freqs:
                token_freqs = all_token_freqs[i]
                assert len(tokens) == len(token_freqs)
                zip_list.append(token_freqs)
            features = list(zip(*zip_list))
            all_features.append(features)

        return all_tokens, all_masks, all_features
        
        
    def tokenize_single(self, sentence, field_name=None):

        if not sentence:
            return [], []

        sentence = self._detokenize_cjk_chars(sentence)

        if self.force_subword_tokenize:
            tokens = sentence.split(' ')
        else:
            tokens = [t for t in sentence.split(' ') if len(t) > 0]
            if self._preprocess_context and field_name in ('context', 'context_question'):
                tokens = self.preprocess_context(sentence)
 
        if self.force_subword_tokenize:
            return tokens, None

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
            mask = [not self.is_entity(token) and not self.is_device(token) for token in tokens]
            return tokens, mask

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

    def _make_example(self, parts_list, dir_name=None, **kwargs):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        all_ids, all_contexts, all_questions, all_answers = [], [], [], []
        for parts in parts_list:
            _id, sentence, target_code = parts
            question = 'translate from english to thingtalk'
            context = sentence
            answer = target_code
            all_ids.append(self.name + '/' + _id)
            all_contexts.append(context)
            all_questions.append(question)
            all_answers.append(answer)
        return Example.from_raw(all_ids, all_contexts, all_questions, all_answers,
                                tokenize=self.tokenize, lower=False)


@register_task('contextual_almond')
class ContextualAlmond(BaseAlmondTask):
    """Contextual Almond semantic parsing task
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def is_contextual(self):
        return True
    
    def _make_example(self, parts_list, dir_name=None, **kwargs):
        all_ids, all_contexts, all_questions, all_answers = [], [], [], []
        for parts in parts_list:
            _id, context, sentence, target_code = parts
            question = sentence
            answer = target_code
            all_ids.append(self.name + '/' + _id)
            all_contexts.append(context)
            all_questions.append(question)
            all_answers.append(answer)
        return Example.from_raw(all_ids, all_contexts, all_questions, all_answers,
                                tokenize=self.tokenize, lower=False)
    
    


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

    def _make_example(self, parts_list, dir_name=None, **kwargs):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        all_ids, all_contexts, all_questions, all_answers = [], [], [], []
        for parts in parts_list:
            _id, sentence, target_code = parts
            question = 'translate from thingtalk to english'
            context = target_code
            answer = sentence
            all_ids.append(self.name + '/' + _id)
            all_contexts.append(context)
            all_questions.append(question)
            all_answers.append(answer)
        return Example.from_raw(all_ids, all_contexts, all_questions, all_answers,
                                tokenize=self.tokenize, lower=False)


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
    
    def _make_example(self, parts_list, dir_name=None, **kwargs):
        all_ids, all_contexts, all_questions, all_answers = [], [], [], []
        for parts in parts_list:
            _id, context, sentence, target_code = parts
            answer = target_code
            question = sentence
            all_ids.append(self.name + '/' + dir_name + '/' + _id)
            all_contexts.append(context)
            all_questions.append(question)
            all_answers.append(answer)
        return Example.from_raw(all_ids, all_contexts, all_questions, all_answers,
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

    def is_contextual(self):
        return True
    
    def _make_example(self, parts_list, dir_name=None, **kwargs):
        all_ids, all_contexts, all_questions, all_answers = [], [], [], []
        for parts in parts_list:
            _id, context, sentence, target_code = parts
            answer = target_code
            question = sentence
            all_ids.append(self.name + '/' + _id)
            all_contexts.append(context)
            all_questions.append(question)
            all_answers.append(answer)
        return Example.from_raw(all_ids, all_contexts, all_questions, all_answers,
                                tokenize=self.tokenize, lower=False)

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

    def _make_example(self, parts_list, dir_name=None, **kwargs):
        # the question is irrelevant for this task
        all_ids, all_contexts, all_questions, all_answers = [], [], [], []
        for parts in parts_list:
            _id, context, sentence, target_code = parts
            answer = sentence
            context = context + ' ' + target_code
            question = 'what should the agent say ?'
            all_ids.append(self.name + '/' + _id)
            all_contexts.append(context)
            all_questions.append(question)
            all_answers.append(answer)
        return Example.from_raw(all_ids, all_contexts, all_questions, all_answers,
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

    def is_contextual(self):
        return True
    
    @property
    def metrics(self):
        return ['em', 'bleu']

    def _make_example(self, parts_list, dir_name=None, **kwargs):
        # the question is irrelevant for this task, and the sentence is intentionally ignored
        all_ids, all_contexts, all_questions, all_answers = [], [], [], []
        for parts in parts_list:
            _id, context, _sentence, target_code = parts
            answer = target_code
            question = 'what should the agent do ?'
            all_ids.append(self.name + '/' + _id)
            all_contexts.append(context)
            all_questions.append(question)
            all_answers.append(answer)
        return Example.from_raw(all_ids, all_contexts, all_questions, all_answers,
                                tokenize=self.tokenize, lower=False)

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
    
    def _make_example(self, parts_list, dir_name, **kwargs):
        all_ids, all_contexts, all_questions, all_answers = [], [], [], []
        for parts in parts_list:
            _id, sentence, target_code = parts
            language = ISO_to_LANG.get(dir_name, 'English').lower()
            if kwargs.get('lang_as_question'):
                question = 'translate from {} to thingtalk'.format(language)
            else:
                question = 'translate from english to thingtalk'
            context = sentence
            answer = target_code
            all_ids.append(self.name + '/' + dir_name + '/' + _id)
            all_contexts.append(context)
            all_questions.append(question)
            all_answers.append(answer)
        return Example.from_raw(all_ids, all_contexts, all_questions, all_answers,
                                tokenize=self.tokenize, lower=False)


@register_task('almond_dialogue_multilingual_nlu')
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

    def _make_example(self, parts_list, dir_name=None, **kwargs):
        all_ids, all_contexts, all_questions, all_answers = [], [], [], []
        for parts in parts_list:
            _id, context, sentence, target_code = parts
            answer = target_code
            question = sentence
            all_ids.append(self.name + '/' + dir_name + '/' + _id)
            all_contexts.append(context)
            all_questions.append(question)
            all_answers.append(answer)
        return Example.from_raw(all_ids, all_contexts, all_questions, all_answers,
                                tokenize=self.tokenize, lower=False)


@register_task('almond_dialogue_multilingual_nlg')
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

    def _make_example(self, parts_list, dir_name=None, **kwargs):
        # the question is irrelevant for this task
        all_ids, all_contexts, all_questions, all_answers = [], [], [], []
        for parts in parts_list:
            _id, context, sentence, target_code = parts
            context = context + ' ' + target_code
            answer = sentence
            question = 'what should the agent say ?'
            all_ids.append(self.name + '/' + dir_name + '/' + _id)
            all_contexts.append(context)
            all_questions.append(question)
            all_answers.append(answer)
        return Example.from_raw(all_ids, all_contexts, all_questions, all_answers,
                                tokenize=self.tokenize, lower=False)

