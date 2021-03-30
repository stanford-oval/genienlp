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

import functools
import os
import ujson
import numpy as np
import logging
import torch
from bootleg.annotator import Annotator

from .database_utils import is_banned

from bootleg.extract_mentions import extract_mentions
from bootleg.utils.parser_utils import get_full_config
from bootleg import run

from .progbar import progress_bar

logger = logging.getLogger(__name__)


def reverse_bisect_left(a, x, lo=None, hi=None):
    """Find item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.
    """
    if lo is None:
        lo = 0
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x > a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def bootleg_process_examples(ex, bootleg_annotator, args, label, task):
    line = {}
    line['sentence'] = getattr(ex, task.utterance_field())
    
    assert len(label) == 7
    line['cands'] = label[3]
    line['cand_probs'] = list(map(lambda item: list(item), label[4]))
    line['spans'] = label[5]
    line['aliases'] = label[6]
    tokens_type_ids, tokens_type_probs = bootleg_annotator.bootleg.collect_features_per_line(line, args.bootleg_prob_threshold)
    
    if task.utterance_field() == 'question':
        for i in range(len(tokens_type_ids)):
            ex.question_feature[i].type_id = tokens_type_ids[i]
            ex.question_feature[i].type_prob = tokens_type_probs[i]
            ex.context_plus_question_feature[i + len(ex.context.split(' '))].type_id = tokens_type_ids[i]
            ex.context_plus_question_feature[i + len(ex.context.split(' '))].type_prob = tokens_type_probs[i]
    
    else:
        for i in range(len(tokens_type_ids)):
            ex.context_feature[i].type_id = tokens_type_ids[i]
            ex.context_feature[i].type_prob = tokens_type_probs[i]
            ex.context_plus_question_feature[i].type_id = tokens_type_ids[i]
            ex.context_plus_question_feature[i].type_prob = tokens_type_probs[i]
    
    context_plus_question_with_types = task.insert_type_tokens(ex.context_plus_question,
                                                                              ex.context_plus_question_feature,
                                                                              args.add_types_to_text)
    ex = ex._replace(context_plus_question_with_types=context_plus_question_with_types)
    
    return ex


def extract_features_with_annotator(examples, bootleg_annotator, args, task):
    with torch.no_grad():
        bootleg_inputs = []
        for ex in examples:
            bootleg_inputs.append(getattr(ex, task.utterance_field()))
        
        bootleg_labels = bootleg_annotator.label_mentions(bootleg_inputs)
        bootleg_labels_unpacked = list(zip(*bootleg_labels))
        
        for i in range(len(examples)):
            ex = examples[i]
            label = bootleg_labels_unpacked[i]
            examples[i] = bootleg_process_examples(ex, bootleg_annotator, args, label, task)


def init_bootleg_annotator(args, device):
    # instantiate a bootleg object to load config and relevant databases
    bootleg = Bootleg(args)
    bootleg_config = bootleg.create_config(bootleg.fixed_overrides)
    
    # instantiate the annotator class. we use annotator only in server mode
    # for training we use bootleg functions which preprocess and cache data using multiprocessing, and batching to speed up NED
    bootleg_annotator = Annotator(config_args=bootleg_config,
                                  device='cpu' if device.type == 'cpu' else 'cuda',
                                  max_alias_len=args.max_entity_len,
                                  cand_map=bootleg.cand_map,
                                  threshold=args.bootleg_prob_threshold,
                                  progbar_func=functools.partial(progress_bar, disable=True))
    # collect all outputs now; we will filter later
    bootleg_annotator.set_threshold(0.0)
    setattr(bootleg_annotator, 'bootleg', bootleg)
    return bootleg_annotator


def post_process_bootleg_types(qid, type, title, almond_domains):
    # TODO if training on multiple domains (in one run) these mapping should be modified
    # e.g. song is mapped to book which is not correct if training on music domain too
    for domain in almond_domains:
        if domain == 'restaurants':
            # pizzeria
            if qid == 'Q177':
                type = 'Q571'
            
            # ramen
            if qid == 'Q1051265':
                type = 'Q1778821'
            
            if 'cuisine' in title or 'pasta' in title or 'culture of ' in title or \
                    title in ['food', 'type of food or dish', 'dish', 'convenience food', 'rice dish',
                              'dish', 'food ingredient', 'stuffed pasta', 'raw fish dish',
                              'soup', 'country', 'sovereign state', 'noodle', 'intangible cultural heritage']:
                type = 'Q1778821'
            
            elif title in ['city of the United States', 'big city', 'city with millions of inhabitants',
                           'commune of France']:
                type = 'Q2221906'
            
            elif 'restaurant chain' in title or title in ['restaurant', 'food manufacturer']:
                type = 'Q571'
            
            elif 'writer' in title:
                type = 'Q5'
            
            elif title in ['musical group', 'Wikimedia disambiguation page', 'Wikimedia list article', 'film']:
                type = 'unk'
        
        elif domain == 'books':
            if type == 'Q15087423':
                type = 'unk'
            
            # [houghton mifflin award, ciudad de buenos aires award, new berry award]
            if qid in ['Q390074', 'Q1486', 'Q616527']:
                type = 'Q618779'
            
            # [penguin classics, ]
            elif qid in ['Q1336200']:
                type = 'Q57933693'
            
            elif 'book' in title or 'novel' in title or 'poem' in title or title in \
                    ['written work', 'literary work', 'literature', 'play', 'film', 'occurrence', 'song',
                     'fictional human', 'profession',
                     'document', 'day of the week', 'compilation album', 'magazine', 'television series', 'taxon',
                     'Bible translation',
                     'concept', 'disease', 'technique', 'activity', 'food', 'political ideology', 'literary genre',
                     'mountain', 'mental process',
                     'academic discipline', 'base material', 'negative emotion', 'emotion']:
                type = 'Q571'
            elif 'publisher' in title or title in ['editorial collection', 'version, edition, or translation']:
                type = 'Q57933693'
            elif 'person' in title or 'rights activist' in title or 'writer' in title or \
                    title in ['journalist', 'author',
                          'politician',
                          'Esperantist', 'philosopher', 'actor',
                          'painter',
                          'historian', 'lawyer', 'poet', 'singer']:
                type = 'Q5'
            elif title in ['recurring event'] or 'award' in title:
                type = 'Q618779'
            # languages are not in typeid2title of bootleg
            # [language, country, ethnic group, people, republic]
            elif type in ['Q34770', 'Q6256', 'Q41710', 'Q2472587', 'Q7270']:
                type = 'Q315'
            elif title in ['day', 'single', 'musical group', 'English unit of measurement',
                           'Wikimedia disambiguation page', 'Wikimedia list article']:
                type = 'unk'
        
        if domain == 'movies':
            if 'film' in title or title in ['song', 'single', 'media franchise', 'literary work', 'television series',
                                            'written work']:
                type = 'Q11424'
            
            elif 'director' in title:
                type = 'Q3455803'
            
            elif 'genre' in title or 'fiction' in title or title in ['drama', 'comedy']:
                type = 'Q201658'
            
            elif 'producer' in title:
                type = 'Q2500638'
            
            elif 'actor' in title or 'actress' in title:
                type = 'Q33999'
            
            elif 'language' in title or title in ['cinema of country or region']:
                type = 'Q315'
            
            elif 'writer' in title or title in ['composer', 'screenwriter', 'editer', 'singer', 'businessperson',
                                                'playwright',
                                                'art collector', 'comedian', 'musician', 'aircraft pilot',
                                                'philanthropist',
                                                'restaurateur', 'guitarist', 'novelist', 'Wikimedia list article',
                                                'Wikimedia disambiguation page', 'journalist', 'musical group']:
                type = 'unk'
        
        
        elif domain == 'music':
            if title in ['song', 'single', 'musical composition', 'ballad', 'extended play', 'literary work',
                         'television series', 'film', 'play']:
                type = 'Q7366'
            elif 'album' in title or title in []:
                type = 'Q482994'
            elif 'genre' in title or title in ['country', 'music by country or region', 'music term', 'republic',
                                               'ethnic group', 'music scene', 'popular music', 'rock music',
                                               'heavy metal', 'music', 'pop music', 'electronic music', 'music style']:
                type = 'Q188451'
            elif 'person' in title or 'actor' in title or 'musician' in title or \
                    title in ['singer', 'musician', 'songwriter',
                              'composer', 'producer',
                              'singer-songwriter', 'musical group', 'drummer',
                              'writer', 'philanthropist', 'public figure',
                              'poet', 'guitarist', 'rapper', 'painter',
                              'film director', 'dancer', 'screenwriter',
                              'rock band', 'university teacher', 'journalist',
                              'television presenter', 'film producer',
                              'saxophonist', 'music pedagogue',
                              'association football player', 'film score composer',
                              'disc jockey', 'record producer', 'engineer',
                              'human biblical figure', 'big band',
                              'musical duo', 'girl group', 'entrepreneur',
                              'boy band', 'musical ensemble', 'artist',
                              'vocal group', 'heavy metal band',
                              'literary character', 'lawyer', 'lyricist',
                              'baseball player', 'pianist', 'recording artist',
                              'autobiographer', 'fashion designer']:
                type = 'Q5'
            
            elif 'language' in title or title in ['cinema of country or region', 'sovereign state', 'Bantu',
                                                  'Serbo-Croatian',
                                                  'big city', 'Upper Guinea Creoles']:
                type = 'Q315'
            
            elif title in ['Wikimedia disambiguation page', 'Wikimedia list article']:
                type = 'unk'
        
        
        elif domain == 'spotify':
            # rap, rap music, griot
            if qid in ['Q6010', 'Q11401', 'Q511054', 'Q10460904', 'Q4955868']:
                type = 'Q188451'

            if title in ['song', 'single', 'musical composition', 'ballad', 'extended play', 'literary work',
                         'television series', 'film', 'play']:
                type = 'Q7366'
            elif 'album' in title or title in []:
                type = 'Q482994'
            elif 'genre' in title or title in ['country', 'music by country or region', 'music term', 'republic',
                                               'ethnic group', 'music scene', 'music style']:
                type = 'Q188451'
            elif 'person' in title or 'musician' in title or \
                    title in ['singer', 'actor', 'musician', 'songwriter',
                              'composer', 'singer-songwriter', 'musical group', 'drummer',
                              'writer', 'poet', 'guitarist', 'rapper', 'painter',
                              'film director', 'rock band', 'university teacher', 'journalist',
                              'television presenter', 'saxophonist', 'music pedagogue',
                              'association football player', 'disc jockey', 'record producer', 'engineer',
                              'human biblical figure', 'big band', 'musical duo', 'girl group',
                              'boy band', 'musical ensemble', 'artist', 'vocal group', 'heavy metal band',
                              'literary character', 'lawyer', 'lyricist', 'baseball player']:
                type = 'Q5'
            
            elif title in ['video game', 'disease', 'city of the United States', 'taxon',
                           'Wikimedia disambiguation page', 'Wikimedia list article']:
                type = 'unk'
    
    return type


class Bootleg(object):
    
    def __init__(self, args):
        self.args = args
        
        self.model_dir = f'{self.args.database_dir}/{self.args.bootleg_model}'
        self.config_path = f'{self.model_dir}/bootleg_config.json'
        self.cand_map = f'{self.args.database_dir}/wiki_entity_data/entity_mappings/alias2qids_wiki.json'

        self.entity_dir = f'{self.args.database_dir}/wiki_entity_data'
        self.embed_dir = f'{self.args.database_dir}/emb_data/'
        
        self.almond_domains = args.almond_domains
        self.bootleg_post_process_types = args.bootleg_post_process_types
        
        with open(f'{self.args.database_dir}/emb_data/entityQID_to_wikidataTypeQID.json', 'r') as fin:
            self.qid2type = ujson.load(fin)
        with open(f'{self.args.database_dir}/es_material/type2id.json', 'r') as fin:
            self.type2id = ujson.load(fin)

        with open(f'{self.args.database_dir}/emb_data/wikidatatitle_to_typeid_0905.json', 'r') as fin:
            title2typeid = ujson.load(fin)
            self.typeid2title = {v: k for k, v in title2typeid.items()}
        
        self.pretrained_bert = f'{self.args.database_dir}/emb_data/pretrained_bert_models'
        
        self.cur_entity_embed_size = 0
        
        # Mapping between model directory model checkpoint name
        model2checkpoint = {'bootleg_wiki': 'bootleg_model',
                            'bootleg_wiki_types': 'bootleg_types',
                            'bootleg_wiki_mini': 'bootleg_model_mini',
                            'bootleg_wiki_kg': 'bootleg_kg'}
        
        self.ckpt_name = model2checkpoint[self.args.bootleg_model]
        self.model_ckpt_path = os.path.join(self.model_dir, self.ckpt_name + '.pt')

        ngpus_per_node = 0
        if getattr(self.args, 'devices', None):
            ngpus_per_node = len(self.args.devices)
        
        self.fixed_overrides = [
            "--run_config.distributed", str(ngpus_per_node > 1 and args.bootleg_distributed_eval),
            "--run_config.ngpus_per_node", str(ngpus_per_node),
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

    def create_jsonl(self, input_path, examples, utterance_field):
        # create jsonl file for examples
        jsonl_input_path = input_path.rsplit('.', 1)[0] + '.jsonl'
        with open(jsonl_input_path, 'w') as fout:
            for ex in examples:
                fout.write(ujson.dumps({"sentence": getattr(ex, utterance_field)}) + '\n')

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
        tokens_type_ids = [[self.args.ned_features_default_val[0]] * self.args.ned_features_size[0] for _ in range(len(tokenized))]
        tokens_type_probs = [[self.args.ned_features_default_val[1]] * self.args.ned_features_size[1] for _ in range(len(tokenized))]
    
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
            
                padded_type_ids = self.pad_values(type_ids, self.args.ned_features_size[0], self.args.ned_features_default_val[0])
                padded_type_probs = self.pad_values(type_probs, self.args.ned_features_size[1], self.args.ned_features_default_val[1])
            
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
