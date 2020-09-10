from bootleg.annotator import Annotator
from bootleg.utils.parser_utils import get_full_config
from bootleg import run
import logging
import sys
import json
from elasticsearch import exceptions
from .database import ES_RETRY_ATTEMPTS
import time

logger = logging.getLogger(__name__)

class BootlegAnnotator(Annotator):
    def __init__(self, config_args, device='cpu', bootleg_dir=None, max_alias_len=6, cand_map=None, threshold=0.0):
        
        # will be assigned in task
        self.bootleg_es = None
        
        self.bootleg_dir = bootleg_dir
        cand_map = f'{self.bootleg_dir}/entity_db/entity_mappings/alias2qids_wiki.json'
        
        config_args = self.adjust_config_args(config_args)

        super().__init__(config_args, device, max_alias_len, cand_map, threshold)
        
    def adjust_config_args(self, config_args):

        # set the model checkpoint path
        config_args.run_config.init_checkpoint = f'{self.bootleg_dir}/bootleg_wiki/bootleg_model.pt'

        # set the path for the entity db and candidate map
        config_args.data_config.entity_dir = f'{self.bootleg_dir}/entity_db'
        config_args.data_config.alias_cand_map = 'alias2qids_wiki.json'
    
        # set the embedding paths
        config_args.data_config.emb_dir = f'{self.bootleg_dir}/emb_data'
        config_args.data_config.word_embedding.cache_dir = f'{self.bootleg_dir}/emb_data'
    
        for i in range(len(config_args.data_config.ent_embeddings)):
            if config_args.data_config.ent_embeddings[i]['key'] in ['learned', 'avg_title_proj', 'learned_type',
                                                                    'learned_type_wiki', 'learned_type_relations']:
                continue
            config_args.data_config.ent_embeddings[i]['batch_on_the_fly'] = True
        
        return config_args
    
    def return_type_ids(self, tokens):
        ## Read from a file containing output of self.label_mentions(.)
        # import re
        # results = []
        # re_pattern = re.compile('\[.*?\]')
        # with open('dataset/bootleg/almond/multilingual/en/result.txt', 'r', encoding='utf-8') as fin:
        #     for line in fin:
        #         line = line.strip('\n').strip()
        #         parts = re.findall(re_pattern, line)
        #         results.append(parts)
        #
        # result = list(map(lambda part: eval(part), results[i]))

        result = self.label_mentions(' '.join(tokens))
        
        tokens_type_id = [self.bootleg_es.type2id[self.bootleg_es.unk_type]] * len(tokens)
        
        # no mentions found or no labels passed the threshold for the mentions
        if not result:
            return tokens_type_id

        pred_cands, pred_probs, titles, spans, source_aliases = result

        query_temp = json.dumps({"size": 1, "query": {"match": {"value": "{}"}}})
        matches = self.bootleg_es.batch_find_matches(pred_cands, query_temp)
        
        for span, match in zip(spans, matches):
            
            # we don't have that Qid in ES
            if len(match) == 0:
                continue
            # size is 1 (e.g. highest score)
            match = match[0]
            type = match['_source']['type']
            span = span.split(':')
            tokens_type_id[int(span[0]): int(span[1])] = [self.bootleg_es.type2id[type]] * (int(span[1]) - int(span[0]))
            
        return tokens_type_id
