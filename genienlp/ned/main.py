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

import marisa_trie
import ujson

from ..data_utils.almond_utils import quoted_pattern_with_space
from ..ned.ned_utils import has_overlap, is_banned, normalize_text
from ..util import find_span
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
        self._unrecognized_types = set()

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

    def _get_entity_type(self, current_domain, current_function, answer_tokens, string_begin):
        # identify the type of this string based on the surrounding tokens
        type = None
        if string_begin >= 3 and answer_tokens[string_begin - 2] == '(' and answer_tokens[string_begin - 3] == 'Location':
            # new Location ( " ...
            type = 'Location'
        elif string_begin >= 3 and answer_tokens[string_begin - 2] == '(' and answer_tokens[string_begin - 3].startswith('^^'):
            # null ^^com.foo ( " ...
            prefix, suffix = answer_tokens[string_begin - 3].rsplit(':', 1)
            if prefix != '^^tt':
                type = prefix.rsplit('.', 1)[-1] + ' ' + suffix
            else:
                type = suffix

            if type == 'Person' and string_begin >= 6 and answer_tokens[string_begin - 6] in ('director', 'creator', 'actor'):
                # (director|creator|actor) == null ^^org.schema:Person ( " ...
                type = current_domain + ' ' + answer_tokens[string_begin - 6]
        elif string_begin >= 3 and answer_tokens[string_begin - 2] == '=~' and answer_tokens[string_begin - 3] == 'id':
            type = current_domain + ' ' + current_function
        elif (
            string_begin >= 5
            and answer_tokens[string_begin - 2] == '='
            and answer_tokens[string_begin - 3] == 'name'
            and answer_tokens[string_begin - 4] == '('
            and answer_tokens[string_begin - 5].startswith('@')
        ):
            type = current_domain + ' name'
        elif string_begin >= 3 and answer_tokens[string_begin - 2] in ('=', '==', '=~'):
            # foo == " ...
            # foo = "
            type = current_domain + ' ' + answer_tokens[string_begin - 3]
        elif string_begin >= 5 and answer_tokens[string_begin - 5] in ('contains', 'contains~'):
            # contains ( foo , " ...
            type = current_domain + ' ' + answer_tokens[string_begin - 3]
        elif string_begin >= 2 and answer_tokens[string_begin - 2] in (',', '['):
            # in_array ( foo , [ " ...
            # traverse back until reach beginning of list
            i = string_begin - 2
            while answer_tokens[i] != '[' and i > 0:
                i -= 1
            if i > 4 and answer_tokens[i - 4] in ('in_array', 'in_array~'):
                if answer_tokens[i - 2] == 'id':
                    type = current_domain + ' ' + current_function
                else:
                    type = current_domain + ' ' + answer_tokens[i - 2]

        if type:
            type = type.replace('_', ' ')
        return type

    def collect_answer_entity_types(self, tokens, answer):
        entity2type = dict()

        string_begin = None
        answer_tokens = answer.split(' ')
        current_domain = None
        current_function = None

        for i, token in enumerate(answer_tokens):
            if token == '"':
                if string_begin is None:
                    # opening a quoted string
                    string_begin = i + 1
                else:
                    # closing a quoted string
                    if i == string_begin:
                        string_begin = None
                        continue
                    # skip examples with sentence-annotation entity mismatch. hopefully there's not a lot of them.
                    # this is usually caused by paraphrasing where it adds "-" after entity name: "korean-style restaurants"
                    # or add "'" before or after an entity
                    string_end = i
                    entity = answer_tokens[string_begin:string_end]
                    if find_span(tokens, entity) is None:
                        string_begin = None
                        continue

                    type = self._get_entity_type(current_domain, current_function, answer_tokens, string_begin)
                    if type and type.split(' ')[-1] in (
                        'title',
                        'message',
                        'description',
                        'status',
                        'query',
                        'contents',
                        'keyword',
                        'text',
                    ):
                        # free text
                        string_begin = None
                        continue

                    if type:
                        if self.args.ned_normalize_types != 'off':
                            norm_type = self.normalize_types(type)
                        else:
                            norm_type = type
                        if norm_type:
                            assert norm_type in self.type_vocab_to_typeqid, f'{norm_type}, {answer}'
                            entity2type[' '.join(entity)] = norm_type
                            self.all_schema_types.add(norm_type)
                        elif type not in self._unrecognized_types:
                            logger.info("skipped unrecognized type '%s' in %s", type, answer)
                            self._unrecognized_types.add(type)
                    else:
                        logger.warn('could not identify type: %s', answer)
                    string_begin = None
            elif token.startswith('@'):
                current_domain = answer_tokens[i].rsplit('.', 1)[1]
                # parse:
                # @com.foo ( ... ) . bar
                # or
                # @com.foo . bar
                j = i
                if answer_tokens[j + 1] == '(':
                    # we have a device name, skip to the closing parenthesis
                    j += 2
                    while j < len(answer_tokens) and answer_tokens[j] != ')':
                        j += 1
                if j + 2 < len(answer_tokens):
                    current_function = answer_tokens[j + 2]

        return entity2type


class WikiOracleEntityDisambiguator(BaseEntityDisambiguator):
    def __init__(self, args):
        super().__init__(args)
        self._unrecognized_types = set()
        with open(f'{self.args.database_dir}/wiki_entity_data/type_mappings/wiki/qid2typenames.json') as fin:
            self.entityqid2typenames = ujson.load(fin)

    def find_qid2type(self, answer_tokens):
        pass

    def process_examples(self, examples, split_path, utterance_field):
        for n, ex in enumerate(examples):
            if utterance_field == 'question':
                sentence = ex.question
            else:
                sentence = ex.context
            answer = ex.answer

            append_span = []
            answer_qids = quoted_pattern_with_space.findall(answer)
            for qid in answer_qids:
                append_span.append(qid)
                if qid in self.entityqid2typenames and self.entityqid2typenames[qid]:
                    # map entity qid to its types on wikidata
                    append_span.append(' | '.join(self.entityqid2typenames[qid]))
                append_span.append(',')

            if utterance_field == 'question':
                # assert len(tokens_type_ids) == len(tokens_type_probs) == len(tokens_qids) == len(ex.question.split(' '))
                # examples[n].question_feature = features

                # use pad features for non-utterance field
                # examples[n].context_feature = [Entity.get_pad_entity(self.max_features_size)] * len(ex.context.split(' '))

                # override original question with entities added to it
                examples[n].question = sentence + '<e> ' + ' '.join(append_span).strip(' ,') + ' </e>'

            else:
                # assert len(tokens_type_ids) == len(tokens_type_probs) == len(tokens_qids) == len(ex.context.split(' '))
                # examples[n].context_feature = features

                # use pad features for non-utterance field
                # examples[n].question_feature = [Entity.get_pad_entity(self.max_features_size)] * len(ex.question.split(' '))

                # override original context with entities added to it
                examples[n].context = sentence + ' <e> ' + ' '.join(append_span).strip(' ,') + ' </e>'


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
