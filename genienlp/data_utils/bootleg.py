import os
import ujson
import numpy as np
import logging

from .database_utils import is_banned

from bootleg.extract_mentions import extract_mentions
from bootleg.utils.parser_utils import get_full_config
from bootleg import run

from ..util import reverse_bisect_left, post_process_bootleg_types

logger = logging.getLogger(__name__)

class Bootleg(object):
    
    def __init__(self, args):
        self.args = args
        
        self.model_dir = f'{self.args.bootleg_input_dir}/{self.args.bootleg_model}'
        self.config_path = f'{self.model_dir}/bootleg_config.json'
        self.cand_map = f'{self.args.bootleg_input_dir}/wiki_entity_data/entity_mappings/alias2qids_wiki.json'

        self.entity_dir = f'{self.args.bootleg_input_dir}/wiki_entity_data'
        self.embed_dir = f'{self.args.bootleg_input_dir}/emb_data/'
        
        self.almond_domains = args.almond_domains
        self.bootleg_post_process_types = args.bootleg_post_process_types
        
        with open(f'{self.args.bootleg_input_dir}/emb_data/entityQID_to_wikidataTypeQID.json', 'r') as fin:
            self.qid2type = ujson.load(fin)
        with open(f'{self.args.bootleg_input_dir}/es_material/type2id.json', 'r') as fin:
            self.type2id = ujson.load(fin)

        with open(f'{self.args.bootleg_input_dir}/emb_data/wikidatatitle_to_typeid_0905.json', 'r') as fin:
            title2typeid = ujson.load(fin)
            self.typeid2title = {v: k for k, v in title2typeid.items()}
        
        self.pretrained_bert = f'{self.args.bootleg_input_dir}/emb_data/pretrained_bert_models'
        
        self.cur_entity_embed_size = 0
        
        # Mapping between model directory model checkpoint name
        model2checkpoint = {'bootleg_wiki': 'bootleg_model',
                            'bootleg_wiki_types': 'bootleg_types',
                            'bootleg_wiki_mini': 'bootleg_model_mini',
                            'bootleg_wiki_kg': 'bootleg_kg'}
        
        self.ckpt_name = model2checkpoint[self.args.bootleg_model]
        self.model_ckpt_path = os.path.join(self.model_dir, self.ckpt_name + '.pt')

        self.fixed_overrides = [
             "--run_config.timestamp", 'None',
             "--data_config.entity_dir", self.entity_dir,
             "--run_config.eval_batch_size", str(self.args.bootleg_batch_size),
             "--run_config.save_dir", self.args.bootleg_output_dir,
             "--run_config.init_checkpoint", self.model_ckpt_path,
             "--run_config.loglevel", 'debug',
             "--train_config.load_optimizer_from_ckpt", 'False',
             "--data_config.emb_dir", self.embed_dir,
             "--data_config.alias_cand_map", 'alias2qids_wiki.json',
             "--data_config.word_embedding.cache_dir", self.pretrained_bert,
             "--run_config.dataset_threads", str(self.args.bootleg_dataset_threads),
             "--run_config.dataloader_threads", str(self.args.bootleg_dataloader_threads)
        ]
        
    
    def create_config(self, overrides):
        config_args = get_full_config(self.config_path, overrides)
        return config_args

    def create_jsonl(self, input_path, examples, is_contextual):
        # create jsonl file for examples
        jsonl_input_path = input_path.rsplit('.', 1)[0] + '.jsonl'
        with open(jsonl_input_path, 'w') as fout:
            for ex in examples:
                if is_contextual:
                    fout.write(ujson.dumps({"sentence": ex.question}) + '\n')
                else:
                    fout.write(ujson.dumps({"sentence": ex.context}) + '\n')

    def extract_mentions(self, input_path):
        jsonl_input_path = input_path.rsplit('.', 1)[0] + '.jsonl'
        jsonl_output_path = input_path.rsplit('.', 1)[0] + '_bootleg.jsonl'
        extract_mentions(in_filepath=jsonl_input_path, out_filepath=jsonl_output_path,
                         cand_map_file=self.cand_map, max_alias_len=self.args.max_entity_len,
                         num_workers=self.args.bootleg_extract_num_workers)

    def pad_values(self, tokens, max_size, pad_id):
        if len(tokens) > max_size:
            tokens = tokens[:max_size]
        else:
            tokens += [pad_id] * (max_size - len(tokens))
        return tokens

    def disambiguate_mentions(self, config_args):
        run.main(config_args, self.args.bootleg_dump_mode)

    def collect_features_per_line(self, line, threshold):
        
        tokenized = line['sentence'].split(' ')
        tokens_type_ids = [[self.args.features_default_val[0]] * self.args.features_size[0] for _ in range(len(tokenized))]
        tokens_type_probs = [[self.args.features_default_val[1]] * self.args.features_size[1] for _ in range(len(tokenized))]
    
        for alias, all_qids, all_probs, span in zip(line['aliases'], line['cands'], line['cand_probs'], line['spans']):
            # filter qids with confidence lower than a threshold
            idx = reverse_bisect_left(all_probs, threshold)
            all_qids = all_qids[:idx]
            all_probs = all_probs[:idx]

            type_ids = []
            type_probs = []
        
            if not is_banned(alias):
                for qid, prob in zip(all_qids, all_probs):
                    # get all type for a qid
                    if qid in self.qid2type:
                        all_types = self.qid2type[qid]
                    else:
                        all_types = []
                
                    if isinstance(all_types, str):
                        all_types = [all_types]
                
                    if len(all_types):
                        # update
                        # go through all types
                        for type in all_types:
                            if type in self.type2id:
                                title = self.typeid2title.get(type, '?')
                            
                                ## map wikidata types to thingtalk types
                                if self.bootleg_post_process_types:
                                    type = post_process_bootleg_types(qid, type, title, self.almond_domains)
                            
                                type_id = self.type2id[type]
                                type_ids.append(type_id)
                                type_probs.append(prob)
            
                padded_type_ids = self.pad_values(type_ids, self.args.features_size[0], self.args.features_default_val[0])
                padded_type_probs = self.pad_values(type_probs, self.args.features_size[1], self.args.features_default_val[1])
            
                tokens_type_ids[span[0]:span[1]] = [padded_type_ids] * (span[1] - span[0])
                tokens_type_probs[span[0]:span[1]] = [padded_type_probs] * (span[1] - span[0])

        return tokens_type_ids, tokens_type_probs
            
    def collect_features(self, file_name):
        
        all_tokens_type_ids = []
        all_tokens_type_probs = []
        
        threshold = self.args.bootleg_prob_threshold

        with open(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/eval/{self.ckpt_name}/bootleg_labels.jsonl', 'r') as fin:
            for line in fin:
                line = ujson.loads(line)
                tokens_type_ids, tokens_type_probs = self.collect_features_per_line(line, threshold)
                all_tokens_type_ids.append(tokens_type_ids)
                all_tokens_type_probs.append(tokens_type_probs)

        if os.path.exists(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/eval/{self.ckpt_name}/bootleg_embs.npy'):
            with open(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/eval/{self.ckpt_name}/bootleg_embs.npy', 'rb') as fin:
                emb_data = np.load(fin)
                self.cur_entity_embed_size += emb_data.shape[0]
                
        return all_tokens_type_ids, all_tokens_type_probs
    
    
    def merge_embeds(self, file_list):
        all_emb_data = []
        for file_name in file_list:
            emb_file = f'{self.args.bootleg_output_dir}/{file_name}_bootleg/eval/{self.ckpt_name}/bootleg_embs.npy'
            with open(emb_file, 'rb') as fin:
                emb_data = np.load(fin)
                all_emb_data.append(emb_data)

        all_emb_data = np.concatenate(all_emb_data, axis=0)
        
        # add embeddings for the padding and unknown special tokens
        new_emb = np.concatenate([np.zeros([2, all_emb_data.shape[1]], dtype='float'), all_emb_data], axis=0)
        
        os.makedirs(f'{self.args.bootleg_output_dir}/bootleg/eval/{self.ckpt_name}', exist_ok=True)
        np.save(f'{self.args.bootleg_output_dir}/bootleg/eval/{self.ckpt_name}/ent_embedding.npy', new_emb)
