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


import ujson
import logging
import time
from collections import defaultdict

from elasticsearch.client import XPackClient
from elasticsearch.client.utils import NamespacedClient
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch import exceptions
from .database import is_banned
from .database_utils import normalize_text, has_overlap

ES_RETRY_ATTEMPTS = 5

tracer = logging.getLogger('elasticsearch')
tracer.setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

class XPackClientMPCompatible(XPackClient):
    def __getattr__(self, attr_name):
        return NamespacedClient.__getattribute__(self, attr_name)

class RemoteElasticDatabase(object):
    
    def __init__(self, config, type2id, ned_features_default_val, ned_features_size):
        self.type2id = type2id
        self.id2type = {v: k for k, v in self.type2id.items()}
        self.unk_id = ned_features_default_val[0]
        self.unk_type = self.id2type[self.unk_id]
        
        self.ned_features_default_val = ned_features_default_val
        self.ned_features_size = ned_features_size
    
        self.canonical2type = defaultdict(list)
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
        
        # use xpack client that is compatible with multiprocessing
        self.es.xpack = XPackClientMPCompatible(self.es)

    def lookup(self, tokens, min_entity_len=2, max_entity_len=4, answer_entities=None):
    
        if answer_entities is not None:
            tokens_type_ids = self.lookup_entities(tokens, answer_entities)
        else:
            tokens_type_ids = self.single_lookup(tokens, min_entity_len, max_entity_len)
    
        return tokens_type_ids

    def lookup_entities(self, tokens, entities, allow_fuzzy=False):
        tokens_type_ids = [[self.unk_id] * self.ned_features_size[0]] * len(tokens)
        tokens_text = " ".join(tokens)
    
        all_matches = self.batch_find_matches(entities, allow_fuzzy)
    
        for i in range(len(all_matches)):
            match = all_matches[i]
            ent = entities[i]
            assert len(match) in [0, 1]
        
            # sometimes we have tokens like '.' or ',' which receives no match
            if len(match) == 0:
                continue
        
            match = match[0]
            canonical = match['_source']['canonical']
            type = match['_source']['type']
            if not is_banned(canonical) and canonical == ent:
                type = self.type2id.get(type, self.unk_id)
            
                ent_num_tokens = len(ent.split(' '))
                idx = tokens_text.index(ent)
                token_pos = len(tokens_text[:idx].strip().split(' '))
            
                tokens_type_ids[token_pos: token_pos + ent_num_tokens] = [[type] * self.ned_features_size[
                    0]] * ent_num_tokens
    
        return tokens_type_ids

    def batch_find_matches(self, keys_list, allow_fuzzy):
    
        if allow_fuzzy:
            query_temp = ujson.dumps({"size": 1, "query": {
                "multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"], "fuzziness": "AUTO"}}})
        else:
            query_temp = ujson.dumps(
                {"size": 1, "query": {"multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"]}}})
    
        queries = []
        for key in keys_list:
            queries.append(ujson.loads(query_temp.replace('"{}"', '"' + key + '"')))
    
        search_header = ujson.dumps({'index': self.index})
        request = ''
        for q in queries:
            request += '{}\n{}\n'.format(search_header, ujson.dumps(q))
    
        retries = 0
        result = None
        while retries < ES_RETRY_ATTEMPTS:
            try:
                result = self.es.msearch(index=self.index, body=request)
                break
            except (exceptions.ConnectionTimeout, exceptions.ConnectionError, exceptions.TransportError):
                logger.warning('Connection Timed Out!')
                time.sleep(1)
                retries += 1
    
        if not result:
            raise ConnectionError('Could not reconnect to ES database after {} tries...'.format(ES_RETRY_ATTEMPTS))
    
        matches = [res['hits']['hits'] for res in result['responses']]
        return matches

    def single_lookup(self, tokens, min_entity_len, max_entity_len, allow_fuzzy=False):
        # load nltk lazily
        import nltk
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
        tokens_type_ids = [[self.type2id['unk']] * self.ned_features_size[0]] * len(tokens)
    
        max_entity_len = min(max_entity_len, len(tokens))
        min_entity_len = min(min_entity_len, len(tokens))
    
        pos_tagged = nltk.pos_tag(tokens)
        verbs = set([x[0] for x in pos_tagged if x[1].startswith('V')])
    
        used_aliases = []
        gram_attempts = []
        for n in range(max_entity_len, min_entity_len - 1, -1):
            ngrams = nltk.ngrams(tokens, n)
            start = -1
            end = n - 1
            for gram in ngrams:
                start += 1
                end += 1
                gram_text = normalize_text(" ".join(gram))
            
                if not is_banned(gram_text) and not gram_text in verbs:
                    if has_overlap(start, end, used_aliases):
                        continue
                    gram_attempts.append(gram_text)
        
            all_matches = self.batch_find_matches(gram_attempts, allow_fuzzy)
            for i in range(len(all_matches)):
                match = all_matches[i]
                assert len(match) in [0, 1]
            
                # sometimes we have tokens like '.' or ',' which receives no match
                if len(match) == 0:
                    continue
            
                match = match[0]
                canonical = match['_source']['canonical']
                type = match['_source']['type']
                if not is_banned(canonical) and canonical == gram_attempts[i]:
                    used_aliases.append([self.type2id.get(type, self.unk_id), start, end])
    
        for type_id, beg, end in used_aliases:
            tokens_type_ids[beg:end] = [[type_id] * self.ned_features_size[0]] * (end - beg)
    
        return tokens_type_ids

    def batch_lookup(self, tokens_list, allow_fuzzy):
        length = len(tokens_list)
    
        all_currs = [0] * length
        all_lengths = [len(tokens) for tokens in tokens_list]
        all_ends = [len(tokens) for tokens in tokens_list]
        all_tokens_type_ids = [[] for _ in range(length)]
        all_done = [False] * length
    
        while not all(all_done):
            batch_to_lookup = [' '.join(tokens[all_currs[i]:all_ends[i]]) for i, tokens in enumerate(tokens_list)]
        
            all_matches = self.batch_find_matches(batch_to_lookup, allow_fuzzy)
        
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
                if not is_banned(canonical) and canonical == batch_to_lookup[i]:
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
    
    
    def es_dump(self, mode):
        
        assert mode in ['canonical2type', 'type2id'], 'Mode not supported; use either canonical2type or type2id'
    
        if mode == 'canonical2type':
            return_dict = defaultdict(list)
            dump_file_name = 'canonical2type'
        elif mode == 'type2id':
            return_dict = defaultdict(int)
            dump_file_name = 'type2id'

    
        begin = time.time()
        
        def do_search(scroll_id, total_values):
            body = {"size": 10000, "query": {"match_all": {}}}
            
            if scroll_id is None:
                result = self.es.search(index=self.index, body=body, scroll='3m')
            else:
                result = self.es.scroll(scroll_id=scroll_id, scroll='3m')
            total_values += len(result["hits"]["hits"])
            if len(result["hits"]["hits"]) < 10:
                return None, -1
            print("total docs:", total_values)
            for match in result["hits"]["hits"]:
                if mode == 'canonical2type':
                    return_dict[match['_source']['canonical']].append(match['_source']['type'][len('org.wikidata:'):])
                elif mode == 'type2id':
                    if match['_source']['type'] not in return_dict:
                        return_dict[match['_source']['type']] = len(self.type2id)

            scroll_id = result['_scroll_id']
            print('processed: {}, time elapsed: {}'.format(i, time.time() - begin))
            
            return scroll_id, total_values

        scroll_id, total_values = do_search(None, 0)
        i = 0
        chunk = 0
        while True:
            if total_values % 4000000 == 0:
                with open(f'{dump_file_name}_{chunk}.json', 'w') as fout:
                    ujson.dump(return_dict, fout, ensure_ascii=True)
                chunk += 1
                return_dict.clear()
            try:
                scroll_id, total_values = do_search(scroll_id, total_values)
                if scroll_id is None and total_values == -1:
                    break
                i += 1
            except:
                break
        
        # dump any remaining values
        with open(f'{dump_file_name}_{chunk}.json', 'w') as fout:
            ujson.dump(return_dict, fout, ensure_ascii=True)
    