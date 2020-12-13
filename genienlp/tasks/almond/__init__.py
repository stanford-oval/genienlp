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
from collections import defaultdict
import math
import multiprocessing as mp
import ujson
import marisa_trie
import re
from wordfreq import zipf_frequency

from ..base_task import BaseTask
from ..registry import register_task
from ...data_utils.example import Feature
from ...data_utils.database import Database
from ...data_utils.database_utils import DOMAIN_TYPE_MAPPING
from ...data_utils.bootleg import Bootleg
from ..generic_dataset import CQA, context_question_len, token_batch_fn, default_batch_fn
from ...data_utils.example import Example
from .utils import ISO_to_LANG, is_device, is_entity, process_id, is_cjk_char
from ...util import multiwoz_specific_preprocess

from ..base_dataset import Split
from .utils import ISO_to_LANG, is_device, is_entity, process_id, is_cjk_char, process, chunk_file

quoted_pattern_with_space = re.compile(r'\"\s([^"]*?)\s\"')

logger = logging.getLogger(__name__)


class AlmondDataset(CQA):
    """Obtaining dataset for Almond semantic parsing task"""

    base_url = None

    def __init__(self, path, *, make_example, **kwargs):
        
        #TODO fix cache_path for multilingual task
        subsample = kwargs.get('subsample')
        cached_path = kwargs.get('cached_path')
        skip_cache = kwargs.get('skip_cache', True)
        cache_input_data = kwargs.get('cache_input_data', False)
        bootleg = kwargs.get('bootleg', None)
        num_workers = kwargs.get('num_workers', 0)
        features_size = kwargs.get('features_size')
        features_default_val = kwargs.get('features_default_val')
        verbose = kwargs.get('verbose', False)
        
        cache_name = os.path.join(cached_path, os.path.basename(path), str(subsample))
        dir_name = os.path.basename(os.path.dirname(path))

        if os.path.exists(cache_name) and not skip_cache:
            logger.info(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            n = 0
            with open(path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    n += 1

            max_examples = min(n, subsample) if subsample is not None else n
            
            if num_workers > 0:
                num_processes = min(num_workers, int(mp.cpu_count()))
                logger.info(f'Using {num_processes} workers...')
                chunk_size = int(math.ceil(max_examples / num_processes))
                num_chunks = int(math.ceil(max_examples / chunk_size))
    
                base_path, extension = path.rsplit('.', 1)
    
                chunk_file_paths = [f'{base_path}_{chunk_id}.tsv' for chunk_id in range(num_chunks)]
                chunk_file(path, chunk_file_paths, chunk_size, num_chunks)
                num_processes = min(num_processes, num_chunks)
    
                with mp.Pool(processes=num_processes) as pool:
                    process_args = [{'in_file': chunk_file_paths[i], 'chunk_size': chunk_size, 'dir_name': dir_name,
                                     'example_batch_size': 1, 'make_process_example': make_example,
                                     'kwargs': kwargs} for i in range(num_chunks)]
                    results = pool.map(process, process_args)
    
                # merge all results
                examples = [item for sublist in results for item in sublist]
    
                for file in chunk_file_paths:
                    os.remove(file)
            else:
                process_args = {'in_file': path, 'chunk_size': max_examples, 'dir_name': dir_name,
                                'example_batch_size': 1, 'make_process_example': make_example,
                                'kwargs': kwargs}
                examples = process(process_args)
                
            if bootleg:
                config_ovrrides = bootleg.fixed_overrides
                
                input_file_dir = os.path.dirname(path)
                input_file_name = os.path.basename(path.rsplit('.', 1)[0] + '_bootleg.jsonl')
                
                data_overrides = [
                    "--data_config.data_dir", input_file_dir,
                    "--data_config.test_dataset.file", input_file_name
                ]
                
                # get config args
                config_ovrrides.extend(data_overrides)
                config_args = bootleg.create_config(config_ovrrides)
                
                # create jsonl files from input examples
                # jsonl is the input format bootleg expects
                bootleg.create_jsonl(path, examples)
                
                # extract mentions and mention spans in the sentence and write them to output jsonl files
                bootleg.extract_mentions(path)
                
                # find the right entity candidate for each mention
                # extract type ids for each token in input sentence
                all_token_type_ids, all_tokens_type_probs = bootleg.disambiguate_mentions(config_args, input_file_name[:-len('_bootleg.jsonl')])
                
                # override examples features with bootleg features
                assert len(examples) == len(all_token_type_ids) == len(all_tokens_type_probs)
                for n, (ex, tokens_type_ids, tokens_type_probs) in enumerate(zip(examples, all_token_type_ids, all_tokens_type_probs)):
                    if bootleg.is_contextual:
                        for i in range(len(tokens_type_ids)):
                            examples[n].question_feature[i] = ex.question_feature[i]._replace(type_id=tokens_type_ids[i], type_prob=tokens_type_probs[i])
                            examples[n].context_plus_question_feature[i + len(ex.context)] = ex.context_plus_question_feature[i + len(ex.context)]._replace(type_id=tokens_type_ids[i], type_prob=tokens_type_probs[i])
                            
                    else:
                        for i in range(len(tokens_type_ids)):
                            examples[n].context_feature[i] = ex.context_feature[i]._replace(type_id=tokens_type_ids[i], type_prob=tokens_type_probs[i])
                            examples[n].context_plus_question_feature[i] = ex.context_plus_question_feature[i]._replace(type_id=tokens_type_ids[i], type_prob=tokens_type_probs[i])
                            
                if verbose:
                    print()
                    for ex in examples:
                        if bootleg.is_contextual:
                            print(*[f'token: {token}\ttype: {token_type}' for token, token_type in zip(ex.question_tokens, ex.question_feature)], sep='\n')
                        else:
                            print(*[f'token: {token}\ttype: {token_type}' for token, token_type in zip(ex.context, ex.context_feature)], sep='\n')
                            
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
        self._dataset_specific_preprocess = args.almond_dataset_specific_preprocess
        self._almond_has_multiple_programs = args.almond_has_multiple_programs
        
        no_feature_fields = ['answer']
        if self.is_contextual():
            no_feature_fields.append('context')
        else:
            no_feature_fields.append('question')
        self.no_feature_fields = no_feature_fields

        # initialize the database
        self.db = None
        self.bootleg = None
        
        if args.do_ner:
            self.unk_id = args.features_default_val[0]
            self.TTtype2DBtype = dict()
            if self.args.retrieve_method == 'bootleg':
                self._init_bootleg()
            else:
                for domain in self.args.almond_domains:
                    self.TTtype2DBtype.update(DOMAIN_TYPE_MAPPING[domain])
                self._init_db()

    def _init_db(self):
        if self.args.database_type in ['json', 'local-elastic']:
            with open(os.path.join(self.args.database_dir, 'canonical2type.json'), 'r') as fin:
                canonical2type = ujson.load(fin)
            with open(os.path.join(self.args.database_dir, 'type2id.json'), 'r') as fin:
                type2id = ujson.load(fin)
            
            all_canonicals = marisa_trie.Trie(canonical2type.keys())
        
        if self.args.database_type == 'json':
            self.db = Database(canonical2type, type2id, all_canonicals, self.TTtype2DBtype)
        # elif self.args.database_type == 'local-elastic':
        #     self.db = LocalElasticDatabase(db_data_processed)
        # elif self.args.database_type == 'remote-elastic':
        #     self.db = RemoteElasticDatabase(es_config, unk_id, all_aliases, type2id, alias2qid, qid2typeid)
        #     if self.args.create_type_mapping:
        #         es_dump_type2id(self.db)
        #         es_dump_canonical2type(self.db)

    def _init_bootleg(self):
        self.bootleg = Bootleg(self.args, self.is_contextual())

    def is_contextual(self):
        return NotImplementedError
    
    @property
    def metrics(self):
        return ['em', 'sm', 'bleu']

    def _is_program_field(self, field_name):
        raise NotImplementedError()

    def _make_example(self, parts, dir_name, **kwargs):
        raise NotImplementedError()

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, bootleg=self.bootleg, **kwargs)
    
    def _detokenize_cjk_chars(self, sentence):
        output = []
        i = 0
        while i < len(sentence):
            output.append(sentence[i])
            # skip space after cjk chars only if followed by another cjk char
            if is_cjk_char(ord(sentence[i])) and \
                    i+1 < len(sentence) and sentence[i+1] == ' ' and \
                    i+2 < len(sentence) and is_cjk_char(ord(sentence[i+2])):
                i += 2
            else:
                i += 1
        return "".join(output)

    def preprocess_context(self, sentence):
        preprocessed_context = []
        for token in sentence.split(' '):
            if len(token) == 0:
                continue
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
            # (another syntax: ... filter param:geo:Location == location: " XXXX " and ...)
            # ** this should change if annotations convention changes **
        
            # assume first syntax
            idx = answer.index('" ' + ent + ' "')

            schema_entity_type = None

            answer_tokens_after_entity = answer[idx + len('" ' + ent + ' "'):].split()
            if answer_tokens_after_entity[0].startswith('^^'):
                schema_entity_type = answer_tokens_after_entity[0].rsplit(':', 1)[1]
                
            if schema_entity_type is None:
            
                answer_tokens_before_entity = answer[:idx].split()
                
                # check last three tokens to find one that starts with param
                for i in range(3):
                    if answer_tokens_before_entity[-i].startswith('param:'):
                        schema_entity_type = answer_tokens_before_entity[-i]
                        break
            
                if schema_entity_type:
                    schema_entity_type = schema_entity_type.strip('()').rsplit(':', 1)[1]
                    
            if schema_entity_type is None or schema_entity_type not in self.TTtype2DBtype.keys():
                schema_type = self.db.unk_type
            else:
                schema_type = self.TTtype2DBtype[schema_entity_type]
        
            entity2type[ent] = schema_type
    
        return entity2type
    
    def pad_features(self, features, max_size, pad_id):
        if len(features) > max_size:
            tokens = features[:max_size]
        else:
            features += [pad_id] * (max_size - len(features))
        return features

    def oracle_type_ids(self, tokens, entity2type):
        tokens_type_ids = [[self.args.features_default_val[0]] * self.args.features_size[0] for _ in range(len(tokens))]
        tokens_text = " ".join(tokens)

        for ent in entity2type.keys():
            ent_num_tokens = len(ent.split(' '))
            idx = tokens_text.index(ent)
            token_pos = len(tokens_text[:idx].strip().split(' '))
            
            type_id = self.db.type2id[entity2type[ent]]
            type_id = self.pad_features([type_id], self.args.features_size[0], self.args.features_default_val[0])
        
            tokens_type_ids[token_pos: token_pos + ent_num_tokens] = [type_id] * ent_num_tokens
    
        return tokens_type_ids

    def find_type_ids(self, tokens, answer):
        tokens_type_ids = []

        if self.args.database_type == 'json':
            if self.args.retrieve_method == 'naive':
                tokens_type_ids = self.db.lookup(tokens, self.args.lookup_method, self.args.min_entity_len, self.args.max_entity_len)
            elif self.args.retrieve_method == 'entity-oracle':
                answer_entities = quoted_pattern_with_space.findall(answer)
                tokens_type_ids = self.db.lookup(tokens, answer_entities=answer_entities)
            elif self.args.retrieve_method == 'type-oracle':
                entity2type = self.collect_answer_entity_types(answer)
                tokens_type_ids = self.oracle_type_ids(tokens, entity2type)

        return tokens_type_ids
    
    def find_word_freqs(self, tokens, tokens_type_ids):
        token_freqs = []
        
        for token, token_type_id in zip(tokens, tokens_type_ids):
            if token_type_id == self.args.features_default_val[0]:
                token_freqs.append([1.0] * self.args.features_size[1])
            else:
                token_freqs.append([1.0 / (zipf_frequency(token, 'en') + 1e-3)] * self.args.features_size[1])
        return token_freqs

    def find_type_probs(self, tokens, default_val):
        token_freqs = [default_val] * len(tokens)
        return token_freqs

    def tokenize(self, sentence, field_name=None, answer=None):
        if not sentence:
            return [], [], []

        if self.force_subword_tokenize:
            return sentence.split(' '), None
        
        sentence = self._detokenize_cjk_chars(sentence)
        
        if self._dataset_specific_preprocess == 'multiwoz' and self._is_program_field(field_name):
            sentence = multiwoz_specific_preprocess(sentence)
            tokens = [t for t in sentence.split(' ') if len(t) > 0]
        else:
            tokens = [t for t in sentence.split(' ') if len(t) > 0]
            if self._preprocess_context and field_name in ('context'):
                tokens = self.preprocess_context(sentence)

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

        else:
            mask = [not is_entity(token) and not is_device(token) for token in tokens]

        tokens_type_ids, tokens_type_probs, tokens_word_freqs = None, None, None
        
        if 'type_id' in self.args.features:
            tokens_type_ids = [[self.args.features_default_val[0]] * self.args.features_size[0] for _ in range(len(tokens))]
        if 'type_prob' in self.args.features:
            tokens_type_probs = [[self.args.features_default_val[1]] * self.args.features_size[1] for _ in range(len(tokens))]
        if 'word_freq' in self.args.features:
            tokens_word_freqs = [[self.args.features_default_val[2]] * self.args.features_size[2] for _ in range(len(tokens))]

        if self.args.do_ner and self.bootleg is None and field_name not in self.no_feature_fields:
            if 'type_id' in self.args.features:
                tokens_type_ids = self.find_type_ids(tokens, answer)
            if 'type_prob' in self.args.features:
                tokens_type_probs = self.find_type_probs(tokens, self.args.features_size[1])
            if 'word_freq' in self.args.features:
                tokens_word_freqs = self.find_word_freqs(tokens, tokens_type_ids)
                
            if self.args.verbose and self.args.do_ner:
                    print()
                    print(*[f'token: {token}\ttype: {token_type}' for token, token_type in zip(tokens, tokens_type_ids)], sep='\n')
             
        zip_list = []
        if tokens_type_ids:
            assert len(tokens) == len(tokens_type_ids)
            zip_list.append(tokens_type_ids)
        if tokens_type_probs:
            assert len(tokens) == len(tokens_type_probs)
            zip_list.append(tokens_type_probs)
        if tokens_word_freqs:
            assert len(tokens) == len(tokens_word_freqs)
            zip_list.append(tokens_word_freqs)
        features = [Feature(*tup) for tup in zip(*zip_list)]
        
        return tokens, mask, features

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

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        if self._almond_has_multiple_programs:
            _id, sentence, target_code = parts[:3]
        else:
            _id, sentence, target_code = parts
        question = 'translate from english to thingtalk'
        context = sentence
        answer = target_code
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)

@register_task('natural_seq2seq')
class NaturalSeq2Seq(BaseAlmondTask):
    """The Almond seqeunce to sequence task where both sequences are natural language
    i.e. no ThingTalk program. Paraphrasing and translation are examples of this task"""

    def _is_program_field(self, field_name):
        return False

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant
        if len(parts) == 2:
            input_sequence, target_sequence = parts
            _id = "id-null"
        else:
            _id, input_sequence, target_sequence = parts
        question = 'translate from input to output'
        context = input_sequence
        answer = target_sequence
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/natural_seq2seq'), make_example=self._make_example, **kwargs)

    def tokenize(self, sentence, field_name=None):
        if not sentence:
            return [], []

        tokens = [t for t in sentence.split(' ') if len(t) > 0]
        return tokens, None # no mask since it will be ignored

@register_task('contextual_almond')
class ContextualAlmond(BaseAlmondTask):
    """Contextual Almond semantic parsing task
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def is_contextual(self):
        return True

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            _id, context, sentence, target_code = parts[:4]
        else:
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

    def is_contextual(self):
        return False

    def _is_program_field(self, field_name):
        return field_name == 'context'

    def _make_example(self, parts, dir_name=None, **kwargs):
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

    def is_contextual(self):
        return True

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            _id, context, sentence, target_code = parts[:4]
        else:
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

    def is_contextual(self):
        return True

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            _id, context, sentence, target_code = parts[:4]
        else:
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
    and the desired system dialogue act)
    """
    def _is_program_field(self, field_name):
        return field_name == 'context'

    def is_contextual(self):
        return True

    @property
    def metrics(self):
        return ['bleu']

    def _make_example(self, parts, dir_name=None, **kwargs):
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

    def is_contextual(self):
        return True

    @property
    def metrics(self):
        return ['em', 'sm', 'bleu']

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task, and the sentence is intentionally ignored
        _id, context, _sentence, target_code = parts
        question = 'what should the agent do ?'
        context = context
        answer = target_code
        return Example.from_raw(self.name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)
    

class BaseAlmondMultiLingualTask(BaseAlmondTask):
    """ Base task for MultiLingual Almond
    """
    def get_train_processed_ids(self, split):
        all_ids = []
        for ex in split.examples:
            all_ids.append(process_id(ex))
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
                                                         make_example=self._make_example, bootleg=self.bootleg, **kwargs)
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
            sort_key_fn = context_question_len
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
        return ['em', 'sm', 'bleu']
    
    def _make_example(self, parts, dir_name, **kwargs):
        if self._almond_has_multiple_programs:
            _id, sentence, target_code = parts[:3]
        else:
            _id, sentence, target_code = parts
        language = ISO_to_LANG.get(dir_name, 'English').lower()
        if kwargs.get('lang_as_question'):
            question = 'translate from {} to thingtalk'.format(language)
        else:
            question = 'translate from english to thingtalk'
        context = sentence
        answer = target_code
        return Example.from_raw(self.name + '/' + dir_name + '/' + _id, context, question, answer,
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
        return ['em', 'sm', 'bleu']

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            _id, context, sentence, target_code = parts
        else:
            _id, context, sentence, target_code = parts[:4]
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + dir_name + '/' + _id, context, question, answer,
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

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task
        _id, context, sentence, target_code = parts
        question = 'what should the agent say ?'
        context = context + ' ' + target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + dir_name + '/' + _id, context, question, answer,
                                tokenize=self.tokenize, lower=False)

