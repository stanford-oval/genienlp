import logging
import json

from bootleg.annotator import Annotator
from bootleg import run

from bootleg.trainer import Trainer
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.constants import *
from bootleg.utils import utils, logging_utils, data_utils, train_utils, eval_utils
from bootleg.utils.parser_utils import get_full_config
from bootleg.utils.classes.dataset_collection import DatasetCollection
from bootleg.utils.classes.status_reporter import StatusReporter

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
    
    
    def label_file(self, mode, is_writer, logger, world_size=1, rank=0):
        train_utils.setup_train_heads_and_eval_slices(self.args)
        train_utils.setup_run_folders(self.args, mode)
    
    def label_sentence(self, sent):
        result = self.label_mentions(sent)
        return result
        

    def batch_return_type_ids(self, tokens_list):
        
        all_num_mentions = []
        all_tokens_type_id = []
        all_pred_cands = []
        all_spans = []
        query_temp = json.dumps({"size": 1, "query": {"match": {"value": "{}"}}})
        
        for tokens in tokens_list:
            result = self.label_sentence(' '.join(tokens))
            
            # no mentions found or no labels passed the threshold for the mentions
            if not result:
                result = ([], [], [], [], [])

            pred_cands, pred_probs, titles, spans, source_aliases = result
            
            all_spans.append(spans)
            
            # so we know which matches correspond to which sentence
            all_num_mentions.append(len(pred_cands))

            all_pred_cands.extend(pred_cands)
    
            tokens_type_id = [self.bootleg_es.type2id[self.bootleg_es.unk_type]] * len(tokens)
            all_tokens_type_id.append(tokens_type_id)
    
        if len(all_pred_cands) == 0:
            return all_tokens_type_id
        
        all_matches = self.bootleg_es.batch_find_matches(all_pred_cands, query_temp)
        
        curr = 0
        for i in range(len(tokens_list)):
            matches = all_matches[curr:curr+all_num_mentions[i]]
            curr += all_num_mentions[i]
            spans = all_spans[i]
            assert len(spans) == len(matches)
            for span, match in zip(spans, matches):
    
                # we don't have that Qid in ES
                if len(match) == 0:
                    continue
                # size is 1 (e.g. highest score)
                match = match[0]
                type = match['_source']['type']
                span = span.split(':')
                all_tokens_type_id[i][int(span[0]): int(span[1])] = [self.bootleg_es.type2id[type]] * (int(span[1]) - int(span[0]))

        return all_tokens_type_id
