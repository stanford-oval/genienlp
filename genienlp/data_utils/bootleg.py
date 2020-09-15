import os
import logging
import json

from bootleg.annotator import Annotator
from bootleg.utils import train_utils


logger = logging.getLogger(__name__)

class BootlegAnnotator(Annotator):
    def __init__(self, config_args, device='cpu', bootleg_dir=None, max_alias_len=6, cand_map=None, threshold=0.0, debug=False):
        
        # will be assigned in task
        self.bootleg_es = None
        
        self.bootleg_dir = bootleg_dir
        self.debug = debug
        
        if self.debug:
            cand_map = f'{self.bootleg_dir}/test/data/model_training/entity_db/entity_mappings/alias2qids.json'
        else:
            cand_map = f'{self.bootleg_dir}/entity_db/entity_mappings/alias2qids_wiki.json'
        
        
        config_args = self.adjust_config_args(config_args)

        qid2types_file = os.path.join(config_args.data_config.emb_dir, 'wikidata_types_wiki_filt.json')
        
        if qid2types_file is not None:
            with open(qid2types_file, 'r') as fin:
                self.qid2types = json.load(fin)
            self.unk_type_id = max([max(values) for values in self.qid2types.values() if len(values)]) + 1
        else:
            self.qid2types = {}
            self.unk_type_id = self.bootleg_es.type2id[self.bootleg_es.unk_type]

        super().__init__(config_args, device, max_alias_len, cand_map, threshold)
        
    def adjust_config_args(self, config_args):
        
        if self.debug:
    
            # set the model checkpoint path
            config_args.run_config.init_checkpoint = f'{self.bootleg_dir}/runs/test/20200914_114219/model9.pt'
    
            # set the path for the entity db and candidate map
            config_args.data_config.entity_dir = f'{self.bootleg_dir}/test/data/model_training/entity_db'
            config_args.data_config.alias_cand_map = 'alias2qids.json'
    
            # set the data path and RSS500 test file
            config_args.data_config.data_dir = f'{self.bootleg_dir}/nq'
            config_args.data_config.test_dataset.file = 'test_natural_questions_5000.jsonl'
    
            # set the embedding paths
            config_args.data_config.emb_dir = f'{self.bootleg_dir}/test/data/model_training/'
            # config_args.data_config.word_embedding.cache_dir = f'{input_dir}/emb_data'
        
        else:

            # set the model checkpoint path
            config_args.run_config.init_checkpoint = f'{self.bootleg_dir}/bootleg_wiki/bootleg_model.pt'
    
            # set the path for the entity db and candidate map
            config_args.data_config.entity_dir = f'{self.bootleg_dir}/entity_db'
            config_args.data_config.alias_cand_map = 'alias2qids_wiki.json'
    
            # set the embedding paths
            config_args.data_config.emb_dir = f'{self.bootleg_dir}/emb_data'
            config_args.data_config.word_embedding.cache_dir = f'{self.bootleg_dir}/emb_data'
    
            for embedding in config_args.data_config.ent_embeddings:
                if embedding['key'] in ['learned', 'avg_title_proj', 'learned_type', 'learned_type_wiki', 'learned_type_relations']:
                    continue
                embedding['batch_on_the_fly'] = True

        
        return config_args

    def label_file(self, mode, is_writer, logger, world_size=1, rank=0):
        train_utils.setup_train_heads_and_eval_slices(self.args)
        train_utils.setup_run_folders(self.args, mode)
    
    def batch_label_sentences(self, sentences):
        result = self.batch_label_mentions(sentences)
        return result
    
    def label_sentence(self, sent):
        result = self.label_mentions(sent)
        return result
    
    def batch_find_type_ids(self, all_cands):
        result = []
        for qid in all_cands:
            type_ids = self.qid2types.get(qid, None)
            if type_ids is None or len(type_ids) == 0:
                type_id = self.unk_type_id
            else:
                # we always choose first type if qid has multiple types
                type_id = type_ids[0]
            result.append(type_id)
        
        return result
    

    def batch_return_type_ids(self, tokens_list):
    
        result = {'all_pred_cands': [], 'all_pred_probs': [], 'all_titles': [], 'all_spans': [], 'all_source_aliases': []}
        for tokens in tokens_list:
            res = self.label_sentence(' '.join(tokens))
            if res is None:
                res = {'all_pred_cands': [], 'all_pred_probs': [], 'all_titles': [], 'all_spans': [], 'all_source_aliases': []}
            for k, v in res.items():
                result[k].append(v)
        # result = self.batch_label_sentences([' '.join(tokens) for tokens in tokens_list])

        all_tokens_type_id = [[self.unk_type_id] * len(tokens) for tokens in tokens_list]

        all_pred_cands, all_pred_probs, all_titles, all_spans, all_source_aliases = \
            result['all_pred_cands'], result['all_pred_probs'], result['all_titles'], result['all_spans'], result['all_source_aliases']

        # flatten all_pred_cands
        all_num_mentions = [len(pred_cands) for pred_cands in all_pred_cands]
        all_pred_cands_flat = [cand for sublist in all_pred_cands for cand in sublist]

        if len(self.qid2types) == 0:
            query_temp = json.dumps({"size": 1, "query": {"match": {"value": "{}"}}})
            # query ES
            if len(all_pred_cands) == 0:
                return all_tokens_type_id
            all_matches = self.bootleg_es.batch_find_matches(all_pred_cands_flat, query_temp)
            curr = 0
            for i in range(len(tokens_list)):
                matches = all_matches[curr:curr + all_num_mentions[i]]
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
                    all_tokens_type_id[i][int(span[0]): int(span[1])] = [self.bootleg_es.type2id[type]] * \
                                                                        (int(span[1]) - int(span[0]))
            
        else:
            all_type_indices = self.batch_find_type_ids(all_pred_cands_flat)
            curr = 0
            for i in range(len(tokens_list)):
                type_indices = all_type_indices[curr:curr + all_num_mentions[i]]
                curr += all_num_mentions[i]
                spans = all_spans[i]
                assert len(spans) == len(type_indices)
                for span, type_id in zip(spans, type_indices):
                    span = span.split(':')
                    all_tokens_type_id[i][int(span[0]): int(span[1])] = [type_id] * (int(span[1]) - int(span[0]))

        return all_tokens_type_id
