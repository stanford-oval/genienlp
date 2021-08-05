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

import marisa_trie
import ujson

from ..data_utils.almond_utils import quoted_pattern_with_space
from .base import AbstractEntityDisambiguator

logger = logging.getLogger(__name__)


class NaiveEntityDisambiguator(AbstractEntityDisambiguator):
    def __init__(self, args):
        super().__init__(args)

        with open(os.path.join(self.args.database_dir, 'es_material/alias2type.json'), 'r') as fin:
            # alias2type.json is a big file (>4G); load it only when necessary
            self.alias2type = ujson.load(fin)
            all_aliases = marisa_trie.Trie(self.alias2type.keys())
            self.all_aliases = all_aliases

    def find_type_ids(self, tokens, answer=None):
        tokens_type_ids = self.lookup(
            tokens, self.args.database_lookup_method, self.args.min_entity_len, self.args.max_entity_len
        )
        return tokens_type_ids


class EntityOracleEntityDisambiguator(AbstractEntityDisambiguator):
    def __init__(self, args):
        super().__init__(args)

        with open(os.path.join(self.args.database_dir, 'es_material/alias2type.json'), 'r') as fin:
            # alias2type.json is a big file (>4G); load it only when necessary
            self.alias2type = ujson.load(fin)
            all_aliases = marisa_trie.Trie(self.alias2type.keys())
            self.all_aliases = all_aliases

    def find_type_ids(self, tokens, answer):
        answer_entities = quoted_pattern_with_space.findall(answer)
        tokens_type_ids = self.lookup_entities(tokens, answer_entities)

        return tokens_type_ids


class TypeOracleEntityDisambiguator(AbstractEntityDisambiguator):
    def __init__(self, args):
        super().__init__(args)

    def find_type_ids(self, tokens, answer):
        entity2type = self.collect_answer_entity_types(tokens, answer)
        tokens_type_ids = self.oracle_type_ids(tokens, entity2type)
        return tokens_type_ids

    def oracle_type_ids(self, tokens, entity2type):
        tokens_type_ids = [[0] * self.args.max_features_size for _ in range(len(tokens))]
        tokens_text = " ".join(tokens)

        for ent, type in entity2type.items():
            ent_num_tokens = len(ent.split(' '))
            if ent in tokens_text:
                idx = tokens_text.index(ent)
            else:
                logger.warning('Found a mismatch between sentence and annotation entities')
                logger.info(f'sentence: {tokens_text}, entity2type: {entity2type}')
                continue
            token_pos = len(tokens_text[:idx].split())

            typeqid = self.type_vocab_to_typeqid[type]
            type_id = self.typeqid2id[typeqid]
            type_id = self.pad_features([type_id], self.args.max_features_size, 0)

            tokens_type_ids[token_pos : token_pos + ent_num_tokens] = [type_id] * ent_num_tokens

        return tokens_type_ids

    def collect_answer_entity_types(self, tokens, answer):
        entity2type = dict()
        sentence = ' '.join(tokens)

        answer_entities = quoted_pattern_with_space.findall(answer)
        for ent in answer_entities:
            # skip examples with sentence-annotation entity mismatch. hopefully there's not a lot of them.
            # this is usually caused by paraphrasing where it adds "-" after entity name: "korean-style restaurants"
            # or add "'" before or after an entity
            if not re.search(rf'(^|\s){ent}($|\s)', sentence):
                # print(f'***ent: {ent} {tokens} {answer}')
                continue

            # ** this should change if thingtalk syntax changes **

            # ( ... [Book|Music|...] ( ) filter id =~ " position and affirm " ) ...'
            # ... ^^org.schema.Book:Person ( " james clavell " ) ...
            # ... contains~ ( [award|genre|...] , " booker " ) ...
            # ... inLanguage =~ " persian " ...

            # missing syntax from current code
            #  ... @com.spotify . song ( ) filter in_array~ ( id , [ " piano man " , " uptown girl " ] ) ) [ 1 : 2 ]  ...

            # assume first syntax
            idx = answer.index('" ' + ent + ' "')

            type = None
            tokens_before_entity = answer[:idx].split()

            if tokens_before_entity[-2] == 'Location':
                type = 'Location'

            elif tokens_before_entity[-1] in ['>=', '==', '<='] and tokens_before_entity[-2] in [
                'ratingValue',
                'reviewCount',
                'checkoutTime',
                'checkinTime',
            ]:
                type = tokens_before_entity[-2]

            elif tokens_before_entity[-1] == '=~':
                if tokens_before_entity[-2] in ['id', 'value']:
                    # travers previous tokens until find filter
                    i = -3
                    while tokens_before_entity[i] != 'filter' and i - 3 > -len(tokens_before_entity):
                        i -= 1
                    type = tokens_before_entity[i - 3]
                else:
                    type = tokens_before_entity[-2]

            elif tokens_before_entity[-4] == 'contains~':
                type = tokens_before_entity[-2]

            elif tokens_before_entity[-2].startswith('^^'):
                type = tokens_before_entity[-2].rsplit(':', 1)[1]
                if (
                    type == 'Person'
                    and tokens_before_entity[-3] == 'null'
                    and tokens_before_entity[-5] in ['director', 'creator', 'actor']
                ):
                    type = tokens_before_entity[-5]

            elif tokens_before_entity[-1] in [',', '[']:
                type = 'keywords'

            if type:
                # normalize thingtalk types
                type = type.lower()
                for pair in self.wiki2normalized_type:
                    if fnmatch.fnmatch(type, pair[0]):
                        type = pair[1]
                        break

                assert type in self.type_vocab_to_typeqid, f'{type}, {answer}'
            else:
                print(f'{type}, {answer}')
                continue

            entity2type[ent] = type

            self.all_schema_types.add(type)

        return entity2type
