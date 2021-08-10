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
import re

import ujson

from ..data_utils.example import Entity

logger = logging.getLogger(__name__)


class AbstractEntityDisambiguator(object):
    def __init__(self, args):
        self.args = args
        self.max_features_size = self.args.max_features_size

        ### general attributes
        with open(f'{self.args.database_dir}/wiki_entity_data/type_mappings/wiki/type_vocab_to_wikidataqid.json') as fin:
            self.type_vocab_to_typeqid = ujson.load(fin)
            self.typeqid_to_type_vocab = {v: k for k, v in self.type_vocab_to_typeqid.items()}
        with open(f'{self.args.database_dir}/es_material/typeqid2id.json') as fin:
            self.typeqid2id = ujson.load(fin)
            self.id2typeqid = {v: k for k, v in self.typeqid2id.items()}

        #### almond specific attributes
        self.all_schema_types = set()

        # keys are normalized types for each thingtalk property, values are a list of wiki types
        self.almond_type_mapping = dict()

        # a list of tuples: each pair includes a wiki type and their normalized type
        self.wiki2normalized_type = list()

        if self.args.almond_type_mapping_path:
            # read mapping from user-provided file
            with open(os.path.join(self.args.root, self.args.almond_type_mapping_path)) as fin:
                self.almond_type_mapping = ujson.load(fin)
            self.update_wiki2normalized_type()
        else:
            # this file contains mapping between normalized types and wiki types *per domain*
            # we will choose the subset of domains we want via ned_domains
            with open(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database_files/almond_type_mapping.json')
            ) as fin:
                almond_type_mapping_all_domains = ujson.load(fin)
            # only keep subset for provided domains
            for domain in self.args.ned_domains:
                self.almond_type_mapping.update(almond_type_mapping_all_domains[domain])
            self.update_wiki2normalized_type()
        ####

        self.unk_id = 0
        self.unk_type = self.id2typeqid[self.unk_id]

    def update_wiki2normalized_type(self):
        matches, inclusions = [], []
        for normalized_type, titles in self.almond_type_mapping.items():
            for title in titles:
                if re.search(r'[.?*]', title):
                    inclusions.append((title, normalized_type))
                else:
                    matches.append((title, normalized_type))

        # do wildcard matching only after going through all exact matches
        self.wiki2normalized_type.extend(matches)
        self.wiki2normalized_type.extend(inclusions)

    def normalize_types(self, type):
        norm_type = None
        type = type.lower()
        for pair in self.wiki2normalized_type:
            if fnmatch.fnmatch(type, pair[0]):
                norm_type = pair[1]
                break
        return norm_type

    def process_examples(self, examples, split_path, utterance_field):
        # each subclass should implement their own process_examples method
        raise NotImplementedError()

    def pad_features(self, features, max_size, pad_id):
        if len(features) > max_size:
            features = features[:max_size]
        else:
            features += [pad_id] * (max_size - len(features))
        return features

    def convert_entities_to_strings(self, feat):
        final_types = ''
        if 'type_id' in self.args.entity_attributes:
            all_types = ' | '.join(sorted(self.typeqid_to_type_vocab[self.id2typeqid[id]] for id in feat.type_id if id != 0))
            final_types = '( ' + all_types + ' )'
        final_qids = ''
        if 'qid' in self.args.entity_attributes:
            all_qids = ' | '.join(sorted('Q' + str(id) for id in feat.qid if id != -1))
            final_qids = '[ ' + all_qids + ' ]'

        return final_types, final_qids

    def add_entities_to_text(self, sentence, features):
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
                    final_types, final_qids = self.convert_entities_to_strings(feat)
                    final_token += final_types + final_qids + token
                    # concat all entities with the same type
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
                    final_types, final_qids = self.convert_entities_to_strings(feat)
                    all_tokens = []
                    # concat all entities with the same type
                    while i < len(sentence_tokens) and features[i] == feat:
                        all_tokens.append(sentence_tokens[i])
                        i += 1
                    final_token = ' '.join(filter(lambda token: token != '', [*all_tokens, final_types, final_qids, ';']))
                    sentence_plus_types_tokens.append(final_token)
                else:
                    i += 1

            sentence_plus_types_tokens.append('</e>')

        if not sentence_plus_types_tokens:
            return sentence
        else:
            return ' '.join(sentence_plus_types_tokens)

    def replace_features_inplace(self, examples, all_token_type_ids, all_token_type_probs, all_token_qids, utterance_field):
        assert len(examples) == len(all_token_type_ids) == len(all_token_type_probs) == len(all_token_qids)
        for n, (ex, tokens_type_ids, tokens_type_probs, tokens_qids) in enumerate(
            zip(examples, all_token_type_ids, all_token_type_probs, all_token_qids)
        ):
            features = [Entity(*tup) for tup in zip(tokens_type_ids, tokens_type_probs, tokens_qids)]
            if utterance_field == 'question':
                assert len(tokens_type_ids) == len(tokens_type_probs) == len(tokens_qids) == len(ex.question.split(' '))
                examples[n].question_feature = features

                # use pad features for non-utterance field
                examples[n].context_feature = [Entity.get_pad_entity(self.max_features_size)] * len(ex.context.split(' '))

                # override original question with entities added to it
                examples[n].question = self.add_entities_to_text(ex.question, features)

            else:
                assert len(tokens_type_ids) == len(tokens_type_probs) == len(tokens_qids) == len(ex.context.split(' '))
                examples[n].context_feature = features

                # use pad features for non-utterance field
                examples[n].question_feature = [Entity.get_pad_entity(self.max_features_size)] * len(ex.question.split(' '))

                # override original context with entities added to it
                examples[n].context = self.add_entities_to_text(ex.context, features)
