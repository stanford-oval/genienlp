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

from .almond_utils import quoted_pattern_with_space
from .database_utils import has_overlap, is_banned, normalize_text
from .example import Entity

logger = logging.getLogger(__name__)


class AbstractEntityLinker(object):
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
        for normalized_type, titles in self.almond_type_mapping.items():
            for title in titles:
                self.wiki2normalized_type.append((title, normalized_type))

    def process_examples(self, examples, split_path, utterance_field):
        all_token_type_ids, all_token_type_probs, all_token_qids = [], [], []
        for n, ex in enumerate(examples):
            if utterance_field == 'question':
                sentence = ex.question
            else:
                sentence = ex.context
            answer = ex.answer
            tokens = sentence.split(' ')
            length = len(tokens)

            tokens_type_ids = [[0] * self.args.max_features_size for _ in range(length)]
            tokens_type_probs = [[0] * self.args.max_features_size for _ in range(length)]
            token_qids = [[-1] * self.args.max_features_size for _ in range(length)]

            if 'type_id' in self.args.entity_attributes:
                tokens_type_ids = self.find_type_ids(tokens, answer)
            if 'type_prob' in self.args.entity_attributes:
                tokens_type_probs = self.find_type_probs(tokens, 0, self.args.max_features_size)

            all_token_type_ids.append(tokens_type_ids)
            all_token_type_probs.append(tokens_type_probs)
            all_token_qids.append(token_qids)

        self.replace_features_inplace(examples, all_token_type_ids, all_token_type_probs, all_token_qids, utterance_field)

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

    def find_type_probs(self, tokens, default_val, default_size):
        token_freqs = [[default_val] * default_size] * len(tokens)
        return token_freqs

    def lookup_ngrams(self, tokens, min_entity_len, max_entity_len):
        # load nltk lazily
        import nltk

        nltk.download('averaged_perceptron_tagger', quiet=True)

        tokens_type_ids = [[self.unk_id] * self.max_features_size] * len(tokens)

        max_entity_len = min(max_entity_len, len(tokens))
        min_entity_len = min(min_entity_len, len(tokens))

        pos_tagged = nltk.pos_tag(tokens)
        verbs = set([x[0] for x in pos_tagged if x[1].startswith('V')])

        used_aliases = []
        for n in range(max_entity_len, min_entity_len - 1, -1):
            ngrams = nltk.ngrams(tokens, n)
            start = -1
            end = n - 1
            for gram in ngrams:
                start += 1
                end += 1
                gram_text = normalize_text(" ".join(gram))

                if not is_banned(gram_text) and gram_text not in verbs and gram_text in self.all_aliases:
                    if has_overlap(start, end, used_aliases):
                        continue

                    used_aliases.append([self.typeqid2id.get(self.alias2type[gram_text], self.unk_id), start, end])

        for type_id, beg, end in used_aliases:
            tokens_type_ids[beg:end] = [[type_id] * self.max_features_size] * (end - beg)

        return tokens_type_ids

    def lookup_smaller(self, tokens):
        tokens_type_ids = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            # sort by number of tokens so longer keys get matched first
            matched_items = sorted(self.all_aliases.keys(token), key=lambda item: len(item), reverse=True)
            found = False
            for key in matched_items:
                type = self.alias2type[key]
                key_tokenized = key.split()
                cur = i
                j = 0
                while cur < len(tokens) and j < len(key_tokenized):
                    if tokens[cur] != key_tokenized[j]:
                        break
                    j += 1
                    cur += 1

                if j == len(key_tokenized):
                    if is_banned(' '.join(key_tokenized)):
                        continue

                    # match found
                    found = True
                    tokens_type_ids.extend([[self.typeqid2id[type] * self.max_features_size] for _ in range(i, cur)])

                    # move i to current unprocessed position
                    i = cur
                    break

            if not found:
                tokens_type_ids.append([self.unk_id * self.max_features_size])
                i += 1

        return tokens_type_ids

    def lookup_longer(self, tokens):
        i = 0
        tokens_type_ids = []

        length = len(tokens)
        found = False
        while i < length:
            end = length
            while end > i:
                tokens_str = ' '.join(tokens[i:end])
                if tokens_str in self.all_aliases:
                    # match found
                    found = True
                    tokens_type_ids.extend(
                        [[self.typeqid2id[self.alias2type[tokens_str]] * self.max_features_size] for _ in range(i, end)]
                    )
                    # move i to current unprocessed position
                    i = end
                    break
                else:
                    end -= 1
            if not found:
                tokens_type_ids.append([self.unk_id * self.max_features_size])
                i += 1
            found = False

        return tokens_type_ids

    def lookup_entities(self, tokens, entities):
        tokens_type_ids = [[self.unk_id] * self.max_features_size] * len(tokens)
        tokens_text = " ".join(tokens)

        for ent in entities:
            if ent not in self.all_aliases:
                continue
            ent_num_tokens = len(ent.split(' '))
            idx = tokens_text.index(ent)
            token_pos = len(tokens_text[:idx].strip().split(' '))
            type = self.typeqid2id.get(self.alias2type[ent], self.unk_id)
            tokens_type_ids[token_pos : token_pos + ent_num_tokens] = [[type] * self.max_features_size] * ent_num_tokens

        return tokens_type_ids

    def lookup(self, tokens, database_lookup_method=None, min_entity_len=2, max_entity_len=4):
        tokens_type_ids = [[self.unk_id] * self.max_features_size] * len(tokens)
        if database_lookup_method == 'smaller_first':
            tokens_type_ids = self.lookup_smaller(tokens)
        elif database_lookup_method == 'longer_first':
            tokens_type_ids = self.lookup_longer(tokens)
        elif database_lookup_method == 'ngrams':
            tokens_type_ids = self.lookup_ngrams(tokens, min_entity_len, max_entity_len)
        return tokens_type_ids


class NaiveEntityLinker(AbstractEntityLinker):
    def __init__(self, args):
        super().__init__(args)

        with open(os.path.join(self.args.database_dir, 'es_material/alias2type.json'), 'r') as fin:
            # alias2type.json is a big file (>4G); load it only when necessary
            self.alias2type = ujson.load(fin)
            all_aliases = marisa_trie.Trie(self.alias2type.keys())
            self.all_aliases = all_aliases

    def find_type_ids(self, tokens):
        tokens_type_ids = self.lookup(
            tokens, self.args.database_lookup_method, self.args.min_entity_len, self.args.max_entity_len
        )
        return tokens_type_ids


class TypeOracleEntityLinker(AbstractEntityLinker):
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


class EntityOracleEntityLinker(AbstractEntityLinker):
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
