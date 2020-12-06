import ujson
import numpy as np
from bootleg.extract_mentions import extract_mentions
from bootleg.utils.parser_utils import get_full_config
from bootleg import run

from ..util import reverse_bisect_left


class Bootleg(object):
    
    def __init__(self, bootleg_input_dir, bootleg_output_dir, bootleg_model, unk_id, num_workers, is_contextual,
                 bootleg_load_prepped_data, bootleg_dump_mode, bootleg_batch_size, bootleg_integration):
        self.bootleg_input_dir = bootleg_input_dir
        self.bootleg_output_dir = bootleg_output_dir
        self.bootleg_model = bootleg_model
        self.unk_id = int(unk_id)
        self.num_workers = num_workers
        self.is_contextual = is_contextual
        self.bootleg_load_prepped_data = bootleg_load_prepped_data
        self.bootleg_dump_mode = bootleg_dump_mode
        self.bootleg_batch_size = bootleg_batch_size
        self.bootleg_integration = bootleg_integration
        
        self.model = f'{bootleg_input_dir}/{bootleg_model}'
        self.config_path = f'{self.model}/bootleg_config.json'
        self.cand_map = f'{bootleg_input_dir}/wiki_entity_data/entity_mappings/alias2qids_wiki.json'

        self.entity_dir = f'{bootleg_input_dir}/wiki_entity_data'
        self.embed_dir = f'{bootleg_input_dir}/emb_data/'
        
        with open(f'{bootleg_input_dir}/emb_data/entityQID_to_wikidataTypeQID.json', 'r') as fin:
            self.qid2type = ujson.load(fin)

        with open(f'{bootleg_input_dir}/emb_data/wikidataqid_to_bootlegtypeid.json', 'r') as fin:
            self.type2id = ujson.load(fin)
            
        self.pretrained_bert = f'{bootleg_input_dir}/emb_data/pretrained_bert_models'
        
        self.cur_entity_embed_size = 0

        
        self.fixed_overrides = [
             "--data_config.entity_dir", self.entity_dir,
             "--run_config.eval_batch_size", str(self.bootleg_batch_size),
             "--run_config.save_dir", self.bootleg_output_dir,
             "--run_config.init_checkpoint", self.model,
             "--run_config.loglevel", 'debug',
             "--train_config.load_optimizer_from_ckpt", 'False',
             "--data_config.emb_dir", self.embed_dir,
             "--data_config.alias_cand_map", 'alias2qids_wiki.json',
             "--data_config.word_embedding.cache_dir", self.pretrained_bert,
             '--run_config.dataset_threads', '1',
             '--run_config.dataloader_threads', '4'
        ]
        
    
    def create_config(self, overrides):
        config_args = get_full_config(self.config_path, overrides)
        return config_args

    def create_jsonl(self, input_path, examples):
        # create jsonl file for examples
        jsonl_input_path = input_path.rsplit('.', 1)[0] + '.jsonl'
        with open(jsonl_input_path, 'w') as fout:
            for ex in examples:
                if self.is_contextual:
                    fout.write(ujson.dumps({"sentence": ' '.join(ex.question)}) + '\n')
                else:
                    fout.write(ujson.dumps({"sentence": ' '.join(ex.context)}) + '\n')

    def extract_mentions(self, input_path):
        jsonl_input_path = input_path.rsplit('.', 1)[0] + '.jsonl'
        jsonl_output_path = input_path.rsplit('.', 1)[0] + '_bootleg.jsonl'
        if not self.bootleg_load_prepped_data:
            extract_mentions(in_filepath=jsonl_input_path, out_filepath=jsonl_output_path, cand_map_file=self.cand_map, num_workers=self.num_workers)
    
    def pad_features(self, tokens, max_size, pad_id):
        if len(tokens) > max_size:
            tokens = tokens[:max_size]
        else:
            tokens += [pad_id] * (max_size - len(tokens))
        return tokens
    
    def disambiguate_mentions(self, config_args, file_name, type_size, type_default_val):
        if not self.bootleg_load_prepped_data:
            run.main(config_args, self.bootleg_dump_mode)
        
        all_tokens_type_ids = []
        all_tokens_type_probs = []
        
        threshold = 0.3

        with open(f'{self.bootleg_output_dir}/{file_name}_bootleg/eval/{self.bootleg_model}/bootleg_labels.jsonl', 'r') as fin:
            for i, line in enumerate(fin):
                line = ujson.loads(line)
                tokenized = line['sentence'].split(' ')
                tokens_type_ids = [[self.unk_id] * type_size for _ in range(len(tokenized))]
                tokens_type_probs = [[0.0] * type_size for _ in range(len(tokenized))]
        
                if self.bootleg_integration == 2:
                    # we only have ctx_emb_ids for top candidates
                    # TODO  output ctx_emb_ids for all 30 candidates
                    for ctx_emb_id, prob, span in zip(line['ctx_emb_ids'], line['probs'], line['spans']):
                        
                        # account for padding and UNK id added to embedding matrix
                        ctx_emb_id += 2
                        
                        # account for shift in embedding idx when we merge embeddings for all data splits
                        ctx_emb_id += self.cur_entity_embed_size
                        
                        type_ids = [ctx_emb_id]
                        type_probs = [prob]
                
                        padded_type_ids = self.pad_features(type_ids, type_size, type_default_val)
                        padded_type_probs = self.pad_features(type_probs, type_size, 0.0)
                
                        tokens_type_ids[span[0]:span[1]] = [padded_type_ids] * (span[1] - span[0])
                        tokens_type_probs[span[0]:span[1]] = [padded_type_probs] * (span[1] - span[0])
            
                else:
                    for all_qids, all_probs, span in zip(line['cand_qids'], line['cand_probs'], line['spans']):
                        # filter qids with confidence lower than a threshold
                        idx = reverse_bisect_left(all_probs, threshold, lo=0)
                        all_qids = all_qids[:idx]
                        all_probs = all_probs[:idx]
                        
                        #TODO: now we only keep the first type for each qid
                        # extend so we can keep all types and aggregate later
    
                        type_ids = []
                        type_probs = []
                        for qid, prob in zip(all_qids, all_probs):
                            # get all type for a qid
                            all_types = self.qid2type[qid]
                            if len(all_types):
                                # choose only the first type
                                type_id = self.type2id[all_types[0]]
                                type_ids.append(type_id)
                                type_probs.append(prob)
                            
                        padded_type_ids = self.pad_features(type_ids, type_size, type_default_val)
                        padded_type_probs = self.pad_features(type_probs, type_size, 0.0)
                        
                        tokens_type_ids[span[0]:span[1]] = [padded_type_ids] * (span[1] - span[0])
                        tokens_type_probs[span[0]:span[1]] = [padded_type_probs] * (span[1] - span[0])
                        
                all_tokens_type_ids.append(tokens_type_ids)
                all_tokens_type_probs.append(tokens_type_probs)

        if self.bootleg_load_prepped_data:
            with open(f'{self.bootleg_output_dir}/{file_name}_bootleg/eval/{self.bootleg_model}/bootleg_embs.npy', 'rb') as fin:
                emb_data = np.load(fin)
                self.cur_entity_embed_size += emb_data.shape[0]
                
        return all_tokens_type_ids, all_tokens_type_probs
    
    
    def merge_embeds(self, file_list):
        all_emb_data = []
        for file_name in file_list:
            emb_dir = f'{self.bootleg_output_dir}/{file_name}_bootleg/eval/{self.bootleg_model}'
            with open(f'{emb_dir}/bootleg_embs.npy', 'rb') as fin:
                emb_data = np.load(fin)
                all_emb_data.append(emb_data)

        all_emb_data = np.concatenate(all_emb_data, axis=0)
        
        # add embeddings for the padding and unknown special tokens
        new_emb = np.concatenate([np.zeros([2, all_emb_data.shape[1]], dtype='float'), all_emb_data], axis=0)
        
        import os
        os.makedirs(f'{self.bootleg_output_dir}/bootleg/eval/{self.bootleg_model}', exist_ok=True)
        np.save(f'{self.bootleg_output_dir}/bootleg/eval/{self.bootleg_model}/ent_embedding.npy', new_emb)

