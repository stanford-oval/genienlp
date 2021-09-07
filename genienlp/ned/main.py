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
import logging
import os
import re

import marisa_trie
import ujson

from ..data_utils.almond_utils import quoted_pattern_with_space
from ..ned.ned_utils import has_overlap, is_banned, normalize_text
from .abstract import AbstractEntityDisambiguator

logger = logging.getLogger(__name__)


class BaseEntityDisambiguator(AbstractEntityDisambiguator):
    def __init__(self, args):
        super().__init__(args)
        self.alias2qids = marisa_trie.RecordTrie(f"<{'p'*5}")
        self.qid2typeqid = marisa_trie.RecordTrie("<p")

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

            tokens_type_ids = [self.pad_features([], self.max_features_size, 0) for _ in range(length)]
            tokens_type_probs = [self.pad_features([], self.max_features_size, 0) for _ in range(length)]
            token_qids = [self.pad_features([], self.max_features_size, -1) for _ in range(length)]

            if 'type_id' in self.args.entity_attributes:
                tokens_type_ids = self.find_type_ids(tokens, answer)
            if 'type_prob' in self.args.entity_attributes:
                tokens_type_probs = self.find_type_probs(tokens, 0, self.max_features_size)

            all_token_type_ids.append(tokens_type_ids)
            all_token_type_probs.append(tokens_type_probs)
            all_token_qids.append(token_qids)

        self.replace_features_inplace(examples, all_token_type_ids, all_token_type_probs, all_token_qids, utterance_field)

    def find_type_ids(self, tokens, answer):
        # each subclass should implement their own find_type_ids method
        raise NotImplementedError()

    def find_type_probs(self, tokens, default_val, default_size):
        token_freqs = [[default_val] * default_size] * len(tokens)
        return token_freqs

    def process_types_for_alias(self, alias):
        qids = self.alias2qids[alias]
        if isinstance(qids, list):
            assert len(qids) == 1
            qids = qids[0]
        types = []
        for qid in qids:
            qid = qid.decode('utf-8')
            if qid not in self.qid2typeqid or self.qid2typeqid[qid] not in self.typeqid_to_type_vocab:
                continue
            typeqid = self.qid2typeqid[qid]
            type = self.typeqid_to_type_vocab[typeqid]
            new_type = self.normalize_types(type)
            assert new_type in self.type_vocab_to_typeqid, f'{new_type}, {alias}'
            types.append(self.typeqid2id[self.type_vocab_to_typeqid[new_type]])

        return self.pad_features(types, self.max_features_size, 0)


class NaiveEntityDisambiguator(BaseEntityDisambiguator):
    def __init__(self, args):
        super().__init__(args)
        self.alias2qids = marisa_trie.RecordTrie(f"<{'p'*5}").mmap(
            os.path.join(self.args.database_dir, 'es_material/alias2qids.marisa')
        )
        self.qid2typeqid = marisa_trie.RecordTrie("<p").mmap(
            os.path.join(self.args.database_dir, 'es_material/qid2typeqid.marisa')
        )

    def find_type_ids(self, tokens, answer=None):
        tokens_type_ids = self.lookup_ngrams(tokens)
        return tokens_type_ids

    def lookup_ngrams(self, tokens):
        # load nltk lazily
        import nltk

        nltk.download('averaged_perceptron_tagger', quiet=True)

        tokens_type_ids = [[self.unk_id] * self.max_features_size] * len(tokens)

        max_entity_len = min(self.args.max_entity_len, len(tokens))
        min_entity_len = min(self.args.min_entity_len, len(tokens))

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
                alias = normalize_text(" ".join(gram))

                if not is_banned(alias) and alias not in verbs and alias in self.alias2qids:
                    if has_overlap(start, end, used_aliases):
                        continue

                    padded_type_ids = self.process_types_for_alias(alias)
                    used_aliases.append((padded_type_ids, start, end))

        for type_ids, beg, end in used_aliases:
            tokens_type_ids[beg:end] = [type_ids] * (end - beg)

        return tokens_type_ids


class EntityOracleEntityDisambiguator(BaseEntityDisambiguator):
    def __init__(self, args):
        super().__init__(args)
        self.alias2qids = marisa_trie.RecordTrie(f"<{'p'*5}").mmap(
            os.path.join(self.args.database_dir, 'es_material/alias2qids.marisa')
        )
        self.qid2typeqid = marisa_trie.RecordTrie("<p").mmap(
            os.path.join(self.args.database_dir, 'es_material/qid2typeqid.marisa')
        )

    def find_type_ids(self, tokens, answer):
        answer_entities = quoted_pattern_with_space.findall(answer)
        tokens_type_ids = self.lookup_entities(tokens, answer_entities)
        return tokens_type_ids

    def lookup_entities(self, tokens, entities):
        tokens_type_ids = [[self.unk_id] * self.max_features_size] * len(tokens)
        tokens_text = " ".join(tokens)

        for ent in entities:
            if ent not in self.alias2qids:
                continue
            ent_num_tokens = len(ent.split(' '))
            idx = tokens_text.index(ent)
            token_pos = len(tokens_text[:idx].strip().split(' '))
            padded_type_ids = self.process_types_for_alias(ent)
            tokens_type_ids[token_pos : token_pos + ent_num_tokens] = padded_type_ids

        return tokens_type_ids


class EntityAndTypeOracleEntityDisambiguator(BaseEntityDisambiguator):
    def __init__(self, args):
        super().__init__(args)

    def find_type_ids(self, tokens, answer, aliases=None):
        entity2type = self.collect_answer_entity_types(tokens, answer)

        tokens_type_ids = [[0] * self.max_features_size for _ in range(len(tokens))]
        tokens_text = " ".join(tokens)

        for ent, type in entity2type.items():
            # remove entities that were not detected by bootleg
            if aliases and not any(ent == alias for alias in aliases):
                continue
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
            type_id = self.pad_features([type_id], self.max_features_size, 0)

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
                continue

            # ** this should change if thingtalk syntax changes **
            # ( ... [Book|Music|...] ( ) filter id =~ " position and affirm " ) ...'
            # ... ^^org.schema.Book:Person ( " james clavell " ) ...
            # ... contains~ ( [award|genre|...] , " booker " ) ...
            # ... inLanguage =~ " persian " ...

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
                'contentRating',
            ]:
                type = tokens_before_entity[-2]

            elif tokens_before_entity[-1] == '=~':
                if tokens_before_entity[-2] in ['id', 'value']:
                    # travers previous tokens until find filter
                    i = -3
                    while tokens_before_entity[i] != 'filter' and -(i - 3) < len(tokens_before_entity):
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

            elif tokens_before_entity[-1] == ',' and tokens_before_entity[-2] in ['award']:
                type = 'award'

            elif tokens_before_entity[-1] in [',', '[']:
                # traverse back until reach beginning of list
                i = -1
                while tokens_before_entity[i] != '[' and -(i - 2) < len(tokens_before_entity):
                    i -= 1
                if tokens_before_entity[i - 1] == ',':
                    type = tokens_before_entity[i - 2]
                else:
                    type = 'keywords'

            if type:
                norm_type = self.normalize_types(type)
                assert norm_type in self.type_vocab_to_typeqid, f'{norm_type}, {answer}'
            else:
                print(f'{type}, {answer}')
                continue

            entity2type[ent] = norm_type

            self.all_schema_types.add(norm_type)

        return entity2type


class TypeOracleEntityDisambiguator(EntityAndTypeOracleEntityDisambiguator):
    def __init__(self, args):
        super().__init__(args)

    def process_examples(self, examples, split_path, utterance_field):
        all_token_type_ids, all_token_type_probs, all_token_qids = [], [], []

        file_name = os.path.basename(split_path.rsplit('.', 1)[0])
        with open(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/bootleg_wiki/bootleg_labels.jsonl', 'r') as fin:
            for i, line in enumerate(fin):
                if i >= self.args.subsample:
                    break
                bootleg_aliases = ujson.loads(line)['aliases']
                ex = examples[i]
                if utterance_field == 'question':
                    sentence = ex.question
                else:
                    sentence = ex.context
                answer = ex.answer
                tokens = sentence.split(' ')
                length = len(tokens)

                tokens_type_ids = [self.pad_features([], self.max_features_size, 0) for _ in range(length)]
                tokens_type_probs = [self.pad_features([], self.max_features_size, 0) for _ in range(length)]
                token_qids = [self.pad_features([], self.max_features_size, -1) for _ in range(length)]

                if 'type_id' in self.args.entity_attributes:
                    tokens_type_ids = self.find_type_ids(tokens, answer, bootleg_aliases)
                if 'type_prob' in self.args.entity_attributes:
                    tokens_type_probs = self.find_type_probs(tokens, 0, self.max_features_size)

                all_token_type_ids.append(tokens_type_ids)
                all_token_type_probs.append(tokens_type_probs)
                all_token_qids.append(token_qids)

        self.replace_features_inplace(examples, all_token_type_ids, all_token_type_probs, all_token_qids, utterance_field)
