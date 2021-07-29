#
# Copyright (c) 2020-2021 The Board of Trustees of the Leland Stanford Junior University
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
import fnmatch
import logging
import os

import numpy as np
import torch
import ujson
from bootleg.end2end.bootleg_annotator import BootlegAnnotator as Annotator
from bootleg.end2end.extract_mentions import extract_mentions
from bootleg.run import run_model
from bootleg.utils.parser.parser_utils import parse_boot_and_emm_args

from .database_utils import is_banned, reverse_bisect_left

logger = logging.getLogger(__name__)


class Bootleg(object):
    '''
    A wrapper for all functionalities needed from bootleg. It takes care of data preprocessing,
    running examples through bootleg, and overriding examples faetures with the extracted ones
    '''

    def __init__(self, args):
        logger.info('Initializing Bootleg class')

        self.args = args
        self.model_dir = f'{self.args.database_dir}/{self.args.bootleg_model}'
        self.config_path = f'{self.model_dir}/bootleg_config.yaml'
        self.cand_map = f'{self.args.database_dir}/wiki_entity_data/entity_mappings/alias2qids.json'

        self.entity_dir = f'{self.args.database_dir}/wiki_entity_data'
        self.embed_dir = f'{self.args.database_dir}/wiki_entity_data'

        self.almond_domains = args.almond_domains
        self.bootleg_post_process_types = args.bootleg_post_process_types

        with open(f'{self.args.database_dir}/wiki_entity_data/type_mappings/wiki/qid2typenames.json') as fin:
            self.qid2typenames = ujson.load(fin)
        with open(f'{self.args.database_dir}/wiki_entity_data/type_mappings/wiki/type_vocab_to_wikidataqid.json') as fin:
            self.type_vocab_to_entityqid = ujson.load(fin)
            self.entityqid_to_type_vocab = {v: k for k, v in self.type_vocab_to_entityqid.items()}
        with open(f'{self.args.database_dir}/es_material/typeqid2id.json') as fin:
            self.typeqid2id = ujson.load(fin)
        self.id2typeqid = {v: k for k, v in self.typeqid2id.items()}

        ##### almond specific
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database_files/almond_type_mapping.json')) as fin:
            self.almond_type_mapping = ujson.load(fin)
        # with open(f'{self.args.database_dir}/es_material/almond_type_mapping_inclusion.json', 'r') as fin:
        #     self.almond_type_mapping_include = ujson.load(fin)

        # only keep subset for provided domains
        self.wiki2normalized_type = dict()
        # self.wiki2normalized_type_match = dict()
        for domain in self.almond_domains:
            mapping = self.almond_type_mapping[domain]
            for normalized_type, titles in mapping.items():
                for title in titles:
                    # if title[0] == title[-1] == '*':
                    #     self.wiki2normalized_type_include[title.strip('*')] = normalized_type
                    # else:
                    self.wiki2normalized_type[title] = normalized_type
            # self.type_mapping_match.update(self.almond_type_mapping_match[domain])
            # self.type_mapping_include.update(self.almond_type_mapping_match[domain])

        # self.almond_type2qid = dict()
        # with open(f'{self.args.database_dir}/es_material/almond_type2qid.json', 'r') as fin:
        #     almond_type2qid = ujson.load(fin)
        # for domain in self.almond_domains:
        #     self.almond_type2qid.update(almond_type2qid[domain])
        # self.qid2almond_type = {v: k for k, v in self.almond_type2qid.items()}
        #####

        self.cur_entity_embed_size = 0

        # Mapping between model directory and checkpoint name
        model2checkpoint = {'bootleg_uncased_mini': 'bootleg_wiki.pth', 'bootleg_uncased_super_mini': 'bootleg_wiki.pth'}

        self.ckpt_name, extension = model2checkpoint[self.args.bootleg_model].split('.')
        self.model_ckpt_path = os.path.join(self.model_dir, self.ckpt_name + '.' + extension)

        self.fixed_overrides = [
            # emmental configs
            "--emmental.dataparallel",
            'False',
            "--emmental.log_path",
            self.args.bootleg_output_dir,
            "--emmental.use_exact_log_path",
            'True',
            "--emmental.model_path",
            self.model_ckpt_path,
            "--emmental.device",
            str(getattr(self.args, 'bootleg_device', 0)),
            # run configs
            "--run_config.dataset_threads",
            str(getattr(self.args, 'bootleg_dataset_threads', 1)),
            "--run_config.dataloader_threads",
            str(getattr(self.args, 'bootleg_dataloader_threads', 2)),
            "--run_config.eval_batch_size",
            str(getattr(self.args, 'bootleg_batch_size', 50)),
            "--run_config.log_level",
            'DEBUG',
            # data configs
            "--data_config.print_examples_prep",
            'False',
            "--data_config.entity_dir",
            self.entity_dir,
            "--data_config.entity_prep_dir",
            "prep",
            "--data_config.emb_dir",
            self.embed_dir,
            "--data_config.alias_cand_map",
            'alias2qids.json',
            "--data_config.word_embedding.cache_dir",
            self.args.embeddings,
            "--data_config.print_examples",
            'False',
        ]

    def create_config(self, overrides):
        config_args = parse_boot_and_emm_args(self.config_path, overrides)
        return config_args

    def create_jsonl(self, input_path, examples, utterance_field):
        # create jsonl file for examples
        jsonl_input_path = input_path.rsplit('.', 1)[0] + '.jsonl'
        with open(jsonl_input_path, 'w') as fout:
            for ex in examples:
                fout.write(ujson.dumps({"sentence": getattr(ex, utterance_field)}) + '\n')

    def extract_mentions(self, input_path):
        jsonl_input_path = input_path.rsplit('.', 1)[0] + '.jsonl'
        jsonl_output_path = input_path.rsplit('.', 1)[0] + '_bootleg.jsonl'
        logger.info('Extracting mentions...')
        extract_mentions(
            in_filepath=jsonl_input_path,
            out_filepath=jsonl_output_path,
            cand_map_file=self.cand_map,
            min_alias_len=self.args.min_entity_len,
            max_alias_len=self.args.max_entity_len,
            num_workers=getattr(self.args, 'bootleg_extract_num_workers', 32),
            verbose=False,
        )

    def pad_values(self, tokens, max_size, pad_id):
        if len(tokens) > max_size:
            tokens = tokens[:max_size]
        else:
            tokens += [pad_id] * (max_size - len(tokens))
        return tokens

    def disambiguate_mentions(self, config_args):
        run_model(self.args.bootleg_dump_mode, config_args)

    def post_process_bootleg_types(self, title):
        types = None
        for key in self.wiki2normalized_type.keys():
            if fnmatch.fnmatch(title, key):
                types = self.wiki2normalized_type[key]
        # if title in self.wiki2normalized_type_match:
        #     types = self.wiki2normalized_type_match[title]
        # else:
        #     for key in self.wiki2normalized_type_include.keys():
        #         if key in title:
        #             types = self.wiki2normalized_type_include[key]

        typeqids = None
        if types is not None:
            if isinstance(types, str):
                # typeqids = [self.almond_type2qid[types]]
                typeqids = [self.type_vocab_to_entityqid[types]]
            elif isinstance(types, (list, tuple)):
                typeqids = [self.type_vocab_to_entityqid[type_] for type_ in types]
                # typeqids = [self.almond_type2qid[type_] for type_ in types]

        return typeqids

    def add_type_tokens(self, sentence, features):
        sentence_tokens = sentence.split(' ')
        assert len(sentence_tokens) == len(features)
        sentence_plus_types_tokens = []
        i = 0
        if self.args.add_entities_to_text == 'insert':
            while i < len(sentence_tokens):
                token = sentence_tokens[i]
                feat = features[i]
                # token is an entity
                if any([val != 0 for val in feat.type_id]):
                    final_token = '<e> '
                    final_types = ''
                    if 'type_id' in self.args.entity_attributes:
                        if self.bootleg_post_process_types:
                            all_types = ' | '.join(
                                set(
                                    self.entityqid_to_type_vocab[self.id2typeqid[id]]
                                    for id in feat.type_id
                                    if self.id2typeqid[id] in self.entityqid_to_type_vocab
                                )
                            )
                        else:
                            all_types = ' | '.join(
                                set(
                                    self.entityqid_to_type_vocab[self.id2typeqid[id]]
                                    for id in feat.type_id
                                    if id != self.args.db_unk_id
                                )
                            )
                        final_types = '( ' + all_types + ' ) '
                    final_qids = ''
                    if 'qid' in self.args.entity_attributes:
                        all_qids = ' | '.join(set('Q' + str(id) for id in feat.qid))
                        final_qids = '[' + all_qids + ']'
                    final_token += final_types + final_qids + token
                    # append all entities with same type
                    i += 1
                    while i < len(sentence_tokens) and features[i] == feat:
                        final_token += ' ' + sentence_tokens[i]
                        i += 1
                    final_token += ' </e>'
                    sentence_plus_types_tokens.append(final_token)
                else:
                    sentence_plus_types_tokens.append(token)
                    i += 1

        elif self.args.add_entities_to_text == 'append':
            sentence_plus_types_tokens.extend(sentence_tokens)
            sentence_plus_types_tokens.append('<e>')
            while i < len(sentence_tokens):
                feat = features[i]
                # token is an entity
                if any([val != 0 for val in feat.type_id]):
                    final_types = ''
                    if self.args.add_entities_to_text != 'no':
                        if 'type_id' in self.args.entity_attributes:
                            all_types = ' | '.join(
                                set(
                                    self.entityqid_to_type_vocab[self.id2typeqid[id]]
                                    for id in feat.type_id
                                    if self.id2typeqid[id] in self.entityqid_to_type_vocab
                                )
                            )
                        else:
                            all_types = ' | '.join(
                                set(
                                    self.entityqid_to_type_vocab[self.id2typeqid[id]]
                                    for id in feat.type_id
                                    if id != self.args.db_unk_id
                                )
                            )
                        final_types = ['( ', all_types, ' ) ']
                    final_qids = ''
                    if 'qid' in self.args.entity_attributes:
                        all_qids = ' | '.join(set('Q' + str(id) for id in feat.qid))
                        final_qids = ['[', all_qids, ']']
                    all_tokens = []
                    # append all entities with same type
                    while i < len(sentence_tokens) and features[i] == feat:
                        all_tokens.append(sentence_tokens[i])
                        i += 1
                    final_token = ' '.join([*all_tokens, *final_types, *final_qids, ';'])
                    sentence_plus_types_tokens.append(final_token)
                else:
                    i += 1

            sentence_plus_types_tokens.append('</e>')

        if not sentence_plus_types_tokens:
            return sentence
        else:
            return ' '.join(sentence_plus_types_tokens)

    def collect_features_per_line(self, line, threshold):
        tokenized = line['sentence'].split(' ')
        tokens_type_ids = [[0] * self.args.max_features_size for _ in range(len(tokenized))]
        tokens_type_probs = [[0] * self.args.max_features_size for _ in range(len(tokenized))]
        tokens_qids = [[0] * self.args.max_features_size for _ in range(len(tokenized))]

        for alias, all_qids, all_probs, span in zip(line['aliases'], line['cands'], line['cand_probs'], line['spans']):
            # filter qids with probability lower than a threshold
            idx = reverse_bisect_left(all_probs, threshold)
            all_qids = all_qids[:idx]
            all_probs = all_probs[:idx]

            if len(all_qids) > self.args.max_qids_per_entity:
                all_qids = all_qids[: self.args.max_qids_per_entity]
                all_probs = all_probs[: self.args.max_qids_per_entity]

            type_ids = []
            type_probs = []
            qids = []

            if not is_banned(alias):
                for qid, prob in zip(all_qids, all_probs):
                    # to map qids to unique ids we just need to remove the Q character as qids are distinct
                    qids.append(qid[1:])

                    # get all type for a qid
                    all_types = []
                    if qid in self.qid2typenames and self.qid2typenames[qid]:
                        # map entity qid to its type titles on wikidata ; then map titles to their wikidata qids
                        for typename in self.qid2typenames[qid]:
                            if typename in self.type_vocab_to_entityqid:
                                all_types.append(self.type_vocab_to_entityqid[typename])

                    if len(all_types):
                        # go through all types
                        for typeqid in all_types:
                            if typeqid in self.typeqid2id:
                                # map wikidata types to thingtalk types
                                if self.bootleg_post_process_types:
                                    # map qid to title
                                    title = self.entityqid_to_type_vocab[typeqid]
                                    # process may return multiple types for a single type when it's ambiguous
                                    typeqids = self.post_process_bootleg_types(title)
                                    if typeqids is None:
                                        typeqids = [typeqid]

                                else:
                                    typeqids = [typeqid]

                                for count, typeqid_ in enumerate(typeqids):
                                    if count >= self.args.max_types_per_qid:
                                        break
                                    type_id = self.typeqid2id[typeqid_]
                                    type_ids.append(type_id)
                                    type_probs.append(prob)

                padded_type_ids = self.pad_values(type_ids, self.args.max_features_size, 0)
                padded_type_probs = self.pad_values(type_probs, self.args.max_features_size, 0)
                padded_qids = self.pad_values(qids, self.args.max_features_size, 0)

                tokens_type_ids[span[0] : span[1]] = [padded_type_ids] * (span[1] - span[0])
                tokens_type_probs[span[0] : span[1]] = [padded_type_probs] * (span[1] - span[0])
                tokens_qids[span[0] : span[1]] = [padded_qids] * (span[1] - span[0])

        return tokens_type_ids, tokens_type_probs, tokens_qids

    def process_examples(self, examples, input_file_name, utterance_field):
        # extract features for each token in input sentence from bootleg outputs
        all_token_type_ids, all_tokens_type_probs, all_tokens_qids = self.collect_features(
            input_file_name[: -len('_bootleg.jsonl')]
        )

        all_token_type_ids = all_token_type_ids[: self.args.subsample]
        all_tokens_type_probs = all_tokens_type_probs[: self.args.subsample]
        all_tokens_qids = all_tokens_qids[: self.args.subsample]

        self.replace_features_inplace(examples, all_token_type_ids, all_tokens_type_probs, all_tokens_qids, utterance_field)

    def collect_features(self, file_name):
        all_tokens_type_ids = []
        all_tokens_type_probs = []
        all_tokens_qids = []

        threshold = self.args.bootleg_prob_threshold

        with open(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/{self.ckpt_name}/bootleg_labels.jsonl', 'r') as fin:
            for i, line in enumerate(fin):
                if i >= self.args.subsample:
                    break
                line = ujson.loads(line)
                tokens_type_ids, tokens_type_probs, tokens_qids = self.collect_features_per_line(line, threshold)
                all_tokens_type_ids.append(tokens_type_ids)
                all_tokens_type_probs.append(tokens_type_probs)
                all_tokens_qids.append(tokens_qids)

        if os.path.exists(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/{self.ckpt_name}/bootleg_embs.npy'):
            with open(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/{self.ckpt_name}/bootleg_embs.npy', 'rb') as fin:
                emb_data = np.load(fin)
                self.cur_entity_embed_size += emb_data.shape[0]

        return all_tokens_type_ids, all_tokens_type_probs, all_tokens_qids

    def replace_features_inplace(self, examples, all_token_type_ids, all_tokens_type_probs, all_tokens_qids, utterance_field):
        assert len(examples) == len(all_token_type_ids) == len(all_tokens_type_probs) == len(all_tokens_qids)
        for n, (ex, tokens_type_ids, tokens_type_probs, tokens_qids) in enumerate(
            zip(examples, all_token_type_ids, all_tokens_type_probs, all_tokens_qids)
        ):
            if utterance_field == 'question':
                for i in range(len(tokens_type_ids)):
                    examples[n].question_feature[i].type_id = tokens_type_ids[i]
                    examples[n].question_feature[i].type_prob = tokens_type_probs[i]
                    examples[n].question_feature[i].qid = tokens_qids[i]
                examples[n].question = self.add_type_tokens(ex.question, ex.question_feature)

            else:
                # context is the utterance field
                for i in range(len(tokens_type_ids)):
                    examples[n].context_feature[i].type_id = tokens_type_ids[i]
                    examples[n].context_feature[i].type_prob = tokens_type_probs[i]
                    examples[n].context_feature[i].qid = tokens_qids[i]
                examples[n].context = self.add_type_tokens(ex.context, ex.context_feature)

    def merge_embeds(self, file_list):
        all_emb_data = []
        for file_name in file_list:
            emb_file = f'{self.args.bootleg_output_dir}/{file_name}_bootleg/{self.ckpt_name}/bootleg_embs.npy'
            with open(emb_file, 'rb') as fin:
                emb_data = np.load(fin)
                all_emb_data.append(emb_data)

        all_emb_data = np.concatenate(all_emb_data, axis=0)

        # add embeddings for the padding and unknown special tokens
        new_emb = np.concatenate([np.zeros([2, all_emb_data.shape[1]], dtype='float'), all_emb_data], axis=0)

        os.makedirs(f'{self.args.bootleg_output_dir}/bootleg/eval/{self.ckpt_name}', exist_ok=True)
        np.save(f'{self.args.bootleg_output_dir}/bootleg/eval/{self.ckpt_name}/ent_embedding.npy', new_emb)


class BootlegAnnotator(object):
    '''
    BootlegAnnotator is a wrapper for bootleg's native annotator which takes care of bootleg instantiations and
    extracting required features from annotated examples
    '''

    def __init__(self, args, device, bootleg=None):
        self.args = args
        # instantiate a bootleg object to load config and relevant databases
        if bootleg is None:
            self.bootleg = Bootleg(args)
        else:
            self.bootleg = bootleg
        bootleg_config = self.bootleg.create_config(self.bootleg.fixed_overrides)

        # instantiate the annotator class. we use annotator only in server mode.
        # for training we use bootleg functions which preprocess and cache data using multiprocessing, and batching to speed up NED
        self.annotator = Annotator(
            config=bootleg_config,
            device='cpu' if device.type == 'cpu' else 'cuda',
            min_alias_len=args.min_entity_len,
            max_alias_len=args.max_entity_len,
            cand_map=self.bootleg.cand_map,
            threshold=args.bootleg_prob_threshold,
            model_name=args.bootleg_model,
            verbose=False,
        )
        # collect all outputs now; we will filter later
        self.annotator.set_threshold(0.0)

    def extract_features(self, examples, utterance_field):
        with torch.no_grad():
            bootleg_inputs = []
            for ex in examples:
                bootleg_inputs.append(getattr(ex, utterance_field))

            bootleg_labels = self.annotator.label_mentions(bootleg_inputs)

            keys = tuple(bootleg_labels.keys())
            values = list(bootleg_labels.values())
            values_unpacked = list(zip(*values))

            bootleg_labels_unpacked = [dict(zip(keys, values)) for values in values_unpacked]

            all_token_type_ids, all_tokens_type_probs, all_tokens_qids = [], [], []
            for ex, label in zip(examples, bootleg_labels_unpacked):
                line = {}
                line['sentence'] = getattr(ex, utterance_field)

                assert len(label) == 7
                line['aliases'], line['spans'], line['cands'] = label['aliases'], label['spans'], label['cands']
                line['cand_probs'] = list(map(lambda item: list(item), label['cand_probs']))

                tokens_type_ids, tokens_type_probs, tokens_qids = self.bootleg.collect_features_per_line(
                    line, self.args.bootleg_prob_threshold
                )
                all_token_type_ids.append(tokens_type_ids)
                all_tokens_type_probs.append(tokens_type_probs)
                all_tokens_qids.append(tokens_qids)

            self.bootleg.replace_features_inplace(
                examples, all_token_type_ids, all_tokens_type_probs, all_tokens_qids, utterance_field
            )
