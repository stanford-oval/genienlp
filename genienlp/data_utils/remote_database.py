import ujson
import time
import logging

from elasticsearch.client import XPackClient
from elasticsearch.client.utils import NamespacedClient
from elasticsearch import Elasticsearch, RequestsHttpConnection
from elasticsearch import exceptions
from .database import is_banned

ES_RETRY_ATTEMPTS = 5

tracer = logging.getLogger('elasticsearch')
tracer.setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

class ElasticDatabase(object):
    def __init__(self):
        self.unk_type = 'unk'
        self.unk_id = 0
        self.type2id = {self.unk_type: self.unk_id}
        self.all_aliases = {}
        self.alias2qid = {}
        self.qid2typeid = {}
        self.canonical2type = {}
        self.index = None
        self.es = None
        self.elapsed_time = 0
    
    def batch_es_find_matches(self, keys_list, query_temp):
        
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
                t0 = time.time()
                result = self.es.msearch(index=self.index, body=request)
                t1 = time.time()
                self.elapsed_time += t1 - t0
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

        max_entity_len = min(kwargs.get('max_entity_len', 4), len(tokens))
        min_entity_len = min(kwargs.get('min_entity_len', 2), len(tokens))
        
        # following code is adopted from bootleg's "find_aliases_in_sentence_tag"
        used_aliases = []
        # find largest aliases first
        for n in range(max_entity_len, min_entity_len-1, -1):
            grams = nltk.ngrams(tokens, n)
            j_st = -1
            j_end = n - 1
            gram_attempts = []
            span_begs = []
            span_ends = []
            for gram_words in grams:
                j_st += 1
                j_end += 1
                gram_attempt = get_lnrm(" ".join(gram_words))
                # TODO: remove possessives from alias table
                if len(gram_attempt) > 1:
                    if gram_attempt[-1] == 's' and gram_attempt[-2] == ' ':
                        continue
                # gram_attempts.append(gram_attempt)
                # span_begs.append(j_st)
                # span_ends.append(j_end)
                
                if not is_banned(gram_attempt) and gram_attempt in self.all_aliases:
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
            query_temp = ujson.dumps({"size": 1, "query": {
                "multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"], "fuzziness": "AUTO"}}})
        else:
            query_temp = ujson.dumps(
                {"size": 1, "query": {"multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"]}}})

        max_entity_len = min(kwargs.get('max_entity_len', 4), len(tokens))
        min_entity_len = min(kwargs.get('min_entity_len', 2), len(tokens))

        # following code is adopted from bootleg's "find_aliases_in_sentence_tag"
        used_aliases = []
        # find largest aliases first
        for n in range(max_entity_len, 0, -1):
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
                if not is_banned(canonical) and canonical == gram_attempts[i]:
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
            query_temp = ujson.dumps({"size": 1, "query": {
                "multi_match": {"query": "{}", "fields": ["canonical^8", "aliases^3"], "fuzziness": "AUTO"}}})
        else:
            query_temp = ujson.dumps(
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
        
        from collections import defaultdict
        self.canonical2type = defaultdict(list)

    
    def es_dump_canonical2type(self):
        begin = time.time()
        
        excess_length = len('org.wikidata:')
        
        body = {"size": 10000, "query": {"match_all": {}}}
        result = self.es.search(index=self.index, body=body, scroll='1d')
        print("total docs:", len(result["hits"]["hits"]))
        for match in result["hits"]["hits"]:
            self.canonical2type[match['_source']['canonical']].append(match['_source']['type'][excess_length:])
        
        scroll_id = result['_scroll_id']
        i = 0
        chunk = 0
        total_values = 10000
        while True:
            if total_values % 4000000 == 0:
                with open('dataset/canonical2type_{}.json'.format(chunk), 'w') as fout:
                    ujson.dump(self.canonical2type, fout, ensure_ascii=True)
                chunk += 1
                self.canonical2type = {}
            
            try:
                # if i==3:
                #     break
                result = self.es.scroll(scroll_id=scroll_id, scroll='1d')
                total_values += len(result["hits"]["hits"])
                if len(result["hits"]["hits"]) < 10:
                    break
                print("total docs:", total_values)
                for match in result["hits"]["hits"]:
                    self.canonical2type[match['_source']['canonical']].append(match['_source']['type'][excess_length:])
                scroll_id = result['_scroll_id']
                print('processed: {}, time elapsed: {}'.format(i, time.time() - begin))
                i += 1
            except:
                break
        
        with open('dataset/canonical2type.json', 'w') as fout:
            ujson.dump(self.canonical2type, fout, ensure_ascii=False)
        
        exit(1)
    
    
    def es_dump_type2id(self):
        import time
        begin = time.time()
        
        body = {"size": 10000, "query": {"match_all": {}}}
        result = self.es.search(index=self.index, body=body, scroll='3m')
        print("total docs:", len(result["hits"]["hits"]))
        for match in result["hits"]["hits"]:
            if match['_source']['type'] not in self.type2id:
                self.type2id[match['_source']['type']] = len(self.type2id)
        
        scroll_id = result['_scroll_id']
        i = 0
        total_values = 0
        while True:
            try:
                result = self.es.scroll(scroll_id=scroll_id, scroll='3m')
                total_values += len(result["hits"]["hits"])
                if len(result["hits"]["hits"]) < 10:
                    break
                print("total docs:", total_values)
                for match in result["hits"]["hits"]:
                    if match['_source']['type'] not in self.type2id:
                        self.type2id[match['_source']['type']] = len(self.type2id)
                scroll_id = result['_scroll_id']
                print('processed: {}, time elapsed: {}'.format(i, time.time() - begin))
                i += 1
            except:
                break
        
        with open('dataset/type2id.json', 'w') as fout:
            ujson.dump(self.type2id, fout)
        exit(1)


class LocalElasticDatabase(ElasticDatabase):
    def __init__(self, items):
        super().__init__()
        # TODO fix how types are handled
        TYPES = tuple()
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


