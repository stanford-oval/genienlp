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

from elasticsearch.client import XPackClient
from elasticsearch.client.utils import NamespacedClient
from pytrie import SortedStringTrie as Trie
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch import exceptions


tracer = logging.getLogger('elasticsearch')
tracer.setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


DOMAIN_TYPE_MAPPING = dict()
DOMAIN_TYPE_MAPPING['music'] = {'Person': 'song_artist', 'MusicRecording': 'song_name', 'MusicAlbum': 'song_album'}
DOMAIN_TYPE_MAPPING['spotify'] = {'id': 'song_name', 'song': 'song_name', 'artist': 'song_artist', 'artists': 'song_artist', 'album': 'song_album', 'genres': 'song_genre'}

# Order of types should not be changed (new types can be appended)
TYPES = ('song_name', 'song_artist', 'song_album', 'song_genre')

ES_RETRY_ATTEMPTS = 5

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


BANNED_WORDS = set(
                stopwords.words('english') + \
                ['music', 'musics', 'name', 'names', 'want', 'wants', 'album', 'albums', 'please', 'who', 'show me',
                'play', 'play me', 'plays', 'track', 'tracks', 'song', 'songs', 'record', 'records', 'recordings', 'album', 'something',
                'resume', 'resumes', 'find me', 'the', 'search for me', 'search', 'searches', 'yes', 'yeah', 'popular',
                'release', 'released', 'dance', 'dancing', 'i need', 'i would', ' i will', 'find', 'the list']
                   )

def is_special_case(key):
    if key in BANNED_WORDS:
        return True
    return False


class Database(object):
    def __init__(self, items):
        self.data = Trie(items)
        self.unk_type = self.unk_type
        self.type2id = {self.unk_type:  0}
        self.type2id.update({type: i + 1 for i, type in enumerate(TYPES)})
    
    def update_items(self, new_items, allow_new_types=False):
        new_items_processed = dict()
        for token, type in new_items.items():
            if type in self.type2id.keys():
                new_items_processed[token] = type
            elif allow_new_types:
                new_items_processed[token] = type
                self.type2id[type] = len(self.type2id)
            else:
                # type is unknown
                new_items_processed[token] = self.unk_type
        
        self.data = Trie(new_items_processed)
        
    def lookup_smaller(self, tokens, lookup_dict):
        i = 0
        tokens_type_ids = []
        
        while i < len(tokens):
            token = tokens[i]
            # sort by number of tokens so longer keys get matched first
            matched_items = sorted(lookup_dict.items(prefix=token), key=lambda item: len(item[0]), reverse=True)
            found = False
            for key, type in matched_items:
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
                tokens_type_ids.append(self.type2id[self.unk_type])
                i += 1
                
        return tokens_type_ids

    def lookup_longer(self, tokens, lookup_dict):
        i = 0
        tokens_type_ids = []
        length = len(tokens)
        found = False
        while i < length:
            end = length
            while end > i:
                tokens_str = ' '.join(tokens[i:end])
                if tokens_str in lookup_dict:
                    # match found
                    found = True
                    tokens_type_ids.extend([self.type2id[lookup_dict[tokens_str]] for _ in range(i, end)])
                    # move i to current unprocessed position
                    i = end
                    break
                else:
                    end -= 1
            if not found:
                tokens_type_ids.append(self.type2id[self.unk_type])
                i += 1
            found = False

        return tokens_type_ids

    def lookup(self, tokens, **kwargs):
        
        retrieve_method = kwargs.get('retrieve_method', 'lookup')
        subset = kwargs.get('subset', None)
        lookup_method = kwargs.get('lookup_method', 'longer_first')

        tokens_type_ids = []
        
        if retrieve_method == 'oracle' and subset is not None:
            # types are retrieved from the program
            lookup_dict = Trie(subset)
        else:
            lookup_dict = self.data
            
        if lookup_method == 'smaller_first':
            tokens_type_ids = self.lookup_smaller(tokens, lookup_dict)
        elif lookup_method == 'longer_first':
            tokens_type_ids = self.lookup_longer(tokens, lookup_dict)
 
        return tokens_type_ids


class ElasticDatabase(object):
    def __init__(self):
        self.unk_type = 'unk'
        self.type2id = {self.unk_type: 0}
        self.index = None
        self.es = None

    def batch_find_matches(self, keys_list, query_temp):
        
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


    def batch_lookup(self, tokens_list, **kwargs):
        allow_fuzzy = kwargs.get('allow_fuzzy', False)
        length = len(tokens_list)

        all_currs = [0] * length
        all_lengths = [len(tokens) for tokens in tokens_list]
        all_ends = [len(tokens) for tokens in tokens_list]
        all_tokens_type_ids = [[] for _ in range(length)]
        all_done = [False] * length
        
        while not all(all_done):
            batch_to_lookup = [' '.join(tokens[all_currs[i]:all_ends[i]]) for i, tokens in enumerate(tokens_list)]
            
            if allow_fuzzy:
                query_temp = json.dumps({"size": 1, "query": {"multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"], "fuzziness": "AUTO"}}})
            else:
                query_temp = json.dumps({"size": 1, "query": {"multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"]}}})
            
            all_matches = self.batch_find_matches(batch_to_lookup, query_temp)
            
            for i in range(length):
                if all_done[i]:
                    continue

                match = all_matches[i]
                assert len(match) in [0, 1]
                
                # sometimes we have tokens like '.' or ',' which recieves no match
                if len(match) == 0:
                    all_ends[i] -= 1
                    continue
            
                match = match[0]
                canonical = match['_source']['canonical']
                type = match['_source']['type'][len('org.wikidata:'):]
                if not is_special_case(canonical) and canonical == batch_to_lookup[i]:
                    all_tokens_type_ids[i].extend([self.type2id[type] for _ in range(all_currs[i], all_ends[i])])
                    all_currs[i] = all_ends[i]
                else:
                    all_ends[i] -= 1
                    
            for j in range(length):
                # no matches were found for the span starting from current index
                if all_currs[j] == all_ends[j] and not all_currs[j] >= all_lengths[j]:
                    all_tokens_type_ids[j].append(self.type2id[self.unk_type])
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
    
    def __init__(self, config, type2id):
        super().__init__()
        self.type2id = type2id
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