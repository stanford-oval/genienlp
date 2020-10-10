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

import logging
import json
import time
import unicodedata
import re
import string

from elasticsearch.client import XPackClient
from elasticsearch.client.utils import NamespacedClient
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch import exceptions

from bootleg.extract_mentions import get_lnrm

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

tracer = logging.getLogger('elasticsearch')
tracer.setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

DOMAIN_TYPE_MAPPING = dict()
DOMAIN_TYPE_MAPPING['music'] = {'Person': 'Q5', 'MusicRecording': 'Q7366', 'MusicAlbum': 'Q208569'}   # Q5:human, Q7366:song, Q208569:studio album
DOMAIN_TYPE_MAPPING['spotify'] = {'id': 'Q134556', 'song': 'Q7366', 'artist': 'Q5',
                                  'artists': 'Q5', 'album': 'Q208569', 'genres': 'Q188451'}   # Q188451:music genre

# Order of types should not be changed (new types can be appended)
TYPES = ('song_name', 'song_artist', 'song_album', 'song_genre')

ES_RETRY_ATTEMPTS = 5

BANNED_WORDS = set(
    stopwords.words('english') + \
    ['music', 'musics', 'name', 'names', 'want', 'wants', 'album', 'albums', 'please', 'who', 'show me',
     'play', 'play me', 'plays', 'track', 'tracks', 'song', 'songs', 'record', 'records', 'recordings', 'album',
     'something',
     'resume', 'resumes', 'find me', 'the', 'search for me', 'search', 'searches', 'yes', 'yeah', 'popular',
     'release', 'released', 'dance', 'dancing', 'need', 'i need', 'i would', ' i will', 'find', 'the list', 'get some']
)


def is_special_case(key):
    if key in BANNED_WORDS:
        return True
    return False

def normalize_text(text):
    text = unicodedata.normalize('NFD', text).lower()
    text = re.sub('\s\s+', ' ', text)
    return text

def has_overlap(start, end, used_aliases):
    for alias in used_aliases:
        alias_start, alias_end = alias[1], alias[2]
        if start < alias_end and end > alias_start:
            return True
    return False

class Database(object):
    def __init__(self, canonical2type, type2id, all_canonicals):
        self.canonical2type = canonical2type
        self.type2id = type2id
        self.all_canonicals = all_canonicals

        self.unk_type = 'unk'
        self.unk_id = self.type2id[self.unk_type]

    def lookup_ngrams(self, tokens, max_alias_len):
    
        tokens_type_ids = [self.unk_id] * len(tokens)
    
        max_alias_len = min(max_alias_len, len(tokens))
    
        used_aliases = []
        for n in range(max_alias_len, 0, -1):
            ngrams = nltk.ngrams(tokens, n)
            start = -1
            end = n - 1
            for gram in ngrams:
                start += 1
                end += 1
                gram_text = normalize_text(" ".join(gram))
            
                if not is_special_case(gram_text) and gram_text in self.all_canonicals:
                    if has_overlap(start, end, used_aliases):
                        continue
                
                    used_aliases.append(([self.type2id.get(self.canonical2type[gram_text], self.unk_id), start, end]))
    
        for type_id, beg, end in used_aliases:
            tokens_type_ids[beg:end] = [type_id] * (end - beg)
            
        return tokens_type_ids

    def lookup_smaller(self, tokens):
        
        tokens_type_ids = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            # sort by number of tokens so longer keys get matched first
            matched_items = sorted(self.all_canonicals.keys(token), key=lambda item: len(item), reverse=True)
            found = False
            for key in matched_items:
                type = self.canonical2type[key]
                key_tokenized = key.split()
                cur = i
                j = 0
                while cur < len(tokens) and j < len(key_tokenized):
                    if tokens[cur] != key_tokenized[j]:
                        break
                    j += 1
                    cur += 1
                
                if j == len(key_tokenized):
                    if is_special_case(' '.join(key_tokenized)):
                        continue
                    
                    # match found
                    found = True
                    tokens_type_ids.extend([self.type2id[type] for _ in range(i, cur)])
                    
                    # move i to current unprocessed position
                    i = cur
                    break
            
            if not found:
                tokens_type_ids.append(self.unk_id)
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
                if tokens_str in self.all_canonicals:
                    # match found
                    found = True
                    tokens_type_ids.extend([self.type2id[self.canonical2type[tokens_str]] for _ in range(i, end)])
                    # move i to current unprocessed position
                    i = end
                    break
                else:
                    end -= 1
            if not found:
                tokens_type_ids.append(self.unk_id)
                i += 1
            found = False
        
        return tokens_type_ids

    
    def lookup_entities(self, tokens, entities):
        tokens_type_ids = [self.unk_id] * len(tokens)
        tokens_text = " ".join(tokens)
        
        for ent in entities:
            if ent not in self.all_canonicals:
                continue
            ent_num_tokens = len(ent.split(' '))
            idx = tokens_text.index(ent)
            token_pos = len(tokens_text[:idx].strip().split(' '))
            
            type = self.type2id.get(self.canonical2type[ent], self.unk_id)
            
            tokens_type_ids[token_pos: token_pos+ent_num_tokens] = [type]*ent_num_tokens
        
        return tokens_type_ids
        
    
    def lookup(self, tokens, lookup_method=None, max_alias_len=3, answer_entities=None):
        
        if answer_entities is not None:
            tokens_type_ids = self.lookup_entities(tokens, answer_entities)
        
        if lookup_method == 'smaller_first':
            tokens_type_ids = self.lookup_smaller(tokens)
        elif lookup_method == 'longer_first':
            tokens_type_ids = self.lookup_longer(tokens)
        elif lookup_method == 'ngrams':
            tokens_type_ids = self.lookup_ngrams(tokens, max_alias_len)

        return tokens_type_ids


class ElasticDatabase(object):
    def __init__(self):
        self.unk_type = 'unk'
        self.type2id = {self.unk_type: 0}
        self.unk_id = 0
        self.all_aliases = {}
        self.alias2qid = {}
        self.qid2typeid = {}
        self.canonical2type = {}
        self.index = None
        self.es = None
        self.time = 0
        
        PUNC = string.punctuation
        table = str.maketrans(dict.fromkeys(PUNC))  # OR {key: None for key in string.punctuation}
        self.trans_table = table
    
    def batch_es_find_matches(self, keys_list, query_temp):
        
        queries = []
        for key in keys_list:
            queries.append(json.loads(query_temp.replace('"{}"', '"' + key + '"')))
        
        search_header = json.dumps({'index': self.index})
        request = ''
        for q in queries:
            request += '{}\n{}\n'.format(search_header, json.dumps(q))
        
        retries = 0
        result = None
        while retries < ES_RETRY_ATTEMPTS:
            try:
                t0 = time.time()
                result = self.es.msearch(index=self.index, body=request)
                t1 = time.time()
                self.time += t1 - t0
                break
            except (exceptions.ConnectionTimeout, exceptions.ConnectionError, exceptions.TransportError):
                logger.warning('Connection Timed Out!')
                time.sleep(1)
                retries += 1
        
        if not result:
            raise ConnectionError('Could not reconnect to ES database after {} tries...'.format(ES_RETRY_ATTEMPTS))
        
        matches = [res['hits']['hits'] for res in result['responses']]
        return matches
    
    def single_local_lookup(self, tokens, **kwargs):
        
        tokens_type_ids = [self.unk_id] * len(tokens)
        
        max_alias_len = kwargs.get('max_alias_len', 3)
        max_alias_len = min(max_alias_len, len(tokens))
        
        # following code is adopted from bootleg's "find_aliases_in_sentence_tag"
        used_aliases = []
        # find largest aliases first
        for n in range(max_alias_len, 0, -1):
            grams = nltk.ngrams(tokens, n)
            j_st = -1
            j_end = n - 1
            gram_attempts = []
            span_begs = []
            span_ends = []
            for gram_words in grams:
                j_st += 1
                j_end += 1
                # We don't want punctuation words to be used at the beginning/end
                if len(gram_words[0].translate(self.trans_table).strip()) == 0 or len(
                        gram_words[-1].translate(self.trans_table).strip()) == 0:
                    continue
                gram_attempt = get_lnrm(" ".join(gram_words))
                # TODO: remove possessives from alias table
                if len(gram_attempt) > 1:
                    if gram_attempt[-1] == 's' and gram_attempt[-2] == ' ':
                        continue
                # gram_attempts.append(gram_attempt)
                # span_begs.append(j_st)
                # span_ends.append(j_end)
                
                if not is_special_case(gram_attempt) and gram_attempt in self.all_aliases:
                    keep = True
                    
                    for u_al in used_aliases:
                        u_j_st = u_al[1]
                        u_j_end = u_al[2]
                        if j_st < u_j_end and j_end > u_j_st:
                            keep = False
                            break
                    if not keep:
                        continue
                    
                    used_aliases.append(
                        tuple([self.qid2typeid.get(self.alias2qid[gram_attempt], self.unk_id), j_st, j_end]))
        
        for type_id, beg, end in used_aliases:
            tokens_type_ids[beg:end] = [type_id] * (end - beg)
        return tokens_type_ids
    
    def single_es_lookup(self, tokens, **kwargs):
        
        tokens_type_ids = [self.type2id['unk']] * len(tokens)
        
        allow_fuzzy = kwargs.get('allow_fuzzy', False)
        if allow_fuzzy:
            query_temp = json.dumps({"size": 1, "query": {
                "multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"], "fuzziness": "AUTO"}}})
        else:
            query_temp = json.dumps(
                {"size": 1, "query": {"multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"]}}})
        
        max_alias_len = kwargs.get('max_alias_len', 3)
        max_alias_len = min(max_alias_len, len(tokens))
        
        # following code is adopted from bootleg's "find_aliases_in_sentence_tag"
        used_aliases = []
        # find largest aliases first
        for n in range(max_alias_len, 0, -1):
            grams = nltk.ngrams(tokens, n)
            j_st = -1
            j_end = n - 1
            gram_attempts = []
            span_begs = []
            span_ends = []
            for gram_words in grams:
                j_st += 1
                j_end += 1
                # We don't want punctuation words to be used at the beginning/end
                if len(gram_words[0].translate(self.trans_table).strip()) == 0 or len(
                        gram_words[-1].translate(self.trans_table).strip()) == 0:
                    continue
                gram_attempt = get_lnrm(" ".join(gram_words))
                # TODO: remove possessives from alias table
                if len(gram_attempt) > 1:
                    if gram_attempt[-1] == 's' and gram_attempt[-2] == ' ':
                        continue
                gram_attempts.append(gram_attempt)
                span_begs.append(j_st)
                span_ends.append(j_end)
            
            # all_matches = self.batch_local_find_matches(gram_attempts)
            all_matches = self.batch_es_find_matches(gram_attempts, query_temp)
            # all_matches = [[]] * len(gram_attempts)
            # all_matches[-2] = [{'_source': {'canonical': gram_attempts[-2], 'type': 'org.wikidata:Q101352'}}]
            for i in range(len(all_matches)):
                match = all_matches[i]
                assert len(match) in [0, 1]
                
                # sometimes we have tokens like '.' or ',' which receives no match
                if len(match) == 0:
                    continue
                
                match = match[0]
                canonical = match['_source']['canonical']
                type = match['_source']['type']
                if not is_special_case(canonical) and canonical == gram_attempts[i]:
                    keep = True
                    
                    for u_al in used_aliases:
                        u_j_st = u_al[1]
                        u_j_end = u_al[2]
                        if span_begs[i] < u_j_end and span_ends[i] > u_j_st:
                            keep = False
                            break
                    if not keep:
                        continue
                    
                    used_aliases.append(tuple([type, span_begs[i], span_ends[i]]))
        
        for type, beg, end in used_aliases:
            tokens_type_ids[beg:end] = [self.type2id[type]] * (end - beg)
        return tokens_type_ids
        

    def batch_lookup(self, tokens_list, **kwargs):
        allow_fuzzy = kwargs.get('allow_fuzzy', False)
        length = len(tokens_list)
        
        all_currs = [0] * length
        all_lengths = [len(tokens) for tokens in tokens_list]
        all_ends = [len(tokens) for tokens in tokens_list]
        all_tokens_type_ids = [[] for _ in range(length)]
        all_done = [False] * length
        
        if allow_fuzzy:
            query_temp = json.dumps({"size": 1, "query": {
                "multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"], "fuzziness": "AUTO"}}})
        else:
            query_temp = json.dumps(
                {"size": 1, "query": {"multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"]}}})
        
        while not all(all_done):
            batch_to_lookup = [' '.join(tokens[all_currs[i]:all_ends[i]]) for i, tokens in enumerate(tokens_list)]
            
            all_matches = self.batch_find_matches(batch_to_lookup, query_temp)
            
            for i in range(length):
                if all_done[i]:
                    continue
                
                match = all_matches[i]
                assert len(match) in [0, 1]
                
                # sometimes we have tokens like '.' or ',' which receives no match
                if len(match) == 0:
                    all_ends[i] -= 1
                    continue
                
                match = match[0]
                canonical = match['_source']['canonical']
                type = match['_source']['type']
                if not is_special_case(canonical) and canonical == batch_to_lookup[i]:
                    all_tokens_type_ids[i].extend([self.type2id[type] for _ in range(all_currs[i], all_ends[i])])
                    all_currs[i] = all_ends[i]
                else:
                    all_ends[i] -= 1
            
            for j in range(length):
                # no matches were found for the span starting from current index
                if all_currs[j] == all_ends[j] and not all_currs[j] >= all_lengths[j]:
                    all_tokens_type_ids[j].append(self.type2id['unk'])
                    all_currs[j] += 1
                    all_ends[j] = all_lengths[j]
            
            for j in range(length):
                # reached end of sentence
                if all_currs[j] >= all_lengths[j]:
                    all_done[j] = True
        
        return all_tokens_type_ids


class XPackClientMPCompatible(XPackClient):
    def __getattr__(self, attr_name):
        return NamespacedClient.__getattribute__(self, attr_name)


class RemoteElasticDatabase(ElasticDatabase):
    
    def __init__(self, config, unk_id, all_aliases, type2id, alias2qid, qid2typeid):
        super().__init__()
        self.type2id = type2id
        self.all_aliases = all_aliases
        self.unk_id = unk_id
        self.alias2qid = alias2qid
        self.qid2typeid = qid2typeid
        self._init_db(config)
    
    def _init_db(self, config):
        self.auth = (config['username'], config['password'])
        self.index = config['index']
        self.host = config['host']
        self.es = Elasticsearch(
            hosts=[{'host': self.host, 'port': 443}],
            http_auth=self.auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            maxsize=8
        )
        self.es.xpack = XPackClientMPCompatible(self.es)


class LocalElasticDatabase(ElasticDatabase):
    def __init__(self, items):
        super().__init__()
        self.type2id.update({type: i + 1 for i, type in enumerate(TYPES)})
        self._init_db(items)
    
    def _init_db(self, items):
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        self.index = 'ner'
        
        # create db schema
        schema = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "members": {
                    "properties": {
                        "name": {
                            "type": "text"
                        },
                        "type": {
                            "type": "text"
                        },
                    }
                }
            }
        }
        
        self.es.indices.create(index='db', ignore=400, body=schema)
        
        # add items
        id = 0
        for key, value in items.items():
            self.es.index(index='db', doc_type='music', id=id, body={"name": key, "type": value})
            id += 1