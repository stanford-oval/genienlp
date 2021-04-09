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

import os
import ujson
import numpy as np
import logging
import torch
from .database_utils import is_banned

from bootleg.utils.parser.parser_utils import parse_boot_and_emm_args
from bootleg.run import run_model
from bootleg.end2end.extract_mentions import extract_mentions
from bootleg.end2end.bootleg_annotator import BootlegAnnotator

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
    line['aliases'], line['spans'], line['cands'] = label['aliases'], label['spans'], label['cands']
    line['cand_probs'] = list(map(lambda item: list(item), label['cand_probs']))
    tokens_type_ids, tokens_type_probs = bootleg_annotator.bootleg.collect_features_per_line(line, args.bootleg_prob_threshold)
    
    if task.utterance_field() == 'question':
        for i in range(len(tokens_type_ids)):
            ex.question_feature[i].type_id = tokens_type_ids[i]
            ex.question_feature[i].type_prob = tokens_type_probs[i]

    else:
        for i in range(len(tokens_type_ids)):
            ex.context_feature[i].type_id = tokens_type_ids[i]
            ex.context_feature[i].type_prob = tokens_type_probs[i]

    ex.question_plus_types = task.add_type_tokens(ex.question, ex.question_feature, args.add_types_to_text)
    ex.context_plus_types = task.add_type_tokens(ex.context, ex.context_feature, args.add_types_to_text)
    
    return ex


def extract_features_with_annotator(examples, bootleg_annotator, args, task):
    with torch.no_grad():
        bootleg_inputs = []
        for ex in examples:
            bootleg_inputs.append(getattr(ex, task.utterance_field()))
        
        bootleg_labels = bootleg_annotator.label_mentions(bootleg_inputs)
        
        keys = tuple(bootleg_labels.keys())
        values = list(bootleg_labels.values())
        values_unpacked = list(zip(*values))

        bootleg_labels_unpacked = [dict(zip(keys, values)) for values in values_unpacked]
        
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
    bootleg_annotator = BootlegAnnotator(
                                config=bootleg_config,
                                device='cpu' if device.type == 'cpu' else 'cuda',
                                max_alias_len=args.max_entity_len,
                                cand_map=bootleg.cand_map,
                                threshold=args.bootleg_prob_threshold,
                                model_name=args.bootleg_model,
                                verbose=False
                            )
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
        
        elif domain == 'movies':
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

        elif domain == 'mario':
            if 'artist' in title or 'musician' in title or 'composer' in title or \
                    title in ['singer', 'actor', 'musician', 'songwriter', 'pianist', 'jazz guitarist',
                              'composer', 'singer-songwriter', 'musical group', 'drummer', 'keyboardist',
                              'poet', 'guitarist', 'rapper', 'rock band', 'saxophonist', 'music pedagogue',
                              'opera singer',
                              'disc jockey', 'record producer', 'big band', 'musical duo', 'girl group',
                              'music arranger',
                              'boy band', 'musical ensemble', 'artist', 'lyricist', 'bandleader', 'bassist', 'banjoist',
            
                              # debatable mapping
                              'film actor', 'television actor', 'writer', 'conductor', 'stage actor', 'engineer',
                              'model', 'university teacher',
                              'human biblical figure', 'literary character', 'lawyer', 'lyricist', 'baseball player',
                              'dancer',
                              'screenwriter', 'voice actor', 'film producer', 'politician', 'film director', 'producer',
                              'painter']:
        
                type = ('Q483501', 'Q482980')
    
            elif 'album' in title:
                type = ('Q482994',)
    
            elif 'genre' in title or title in ['country', 'music by country or region', 'music term', 'republic',
                                               'ethnic group', 'music scene', 'music style', 'rock music',
                                               'heavy metal',
                                               'pop music', 'hip hop music', 'electronic music']:
                type = ('Q188451',)
    
            # rap, rap music, griot
            elif qid in ['Q6010', 'Q11401', 'Q511054', 'Q10460904', 'Q4955868']:
                type = ('Q188451',)
    
            elif title in ['song', 'musical composition', 'ballad', 'extended play', 'single',
    
                           # debatable mapping
                           'literary work', 'television series', 'film', 'play']:
        
                type = ('Q7366',)
    
            elif 'cuisine' in title or 'dish' in title or 'pasta' in title or title in ['food']:
                type = ('Q1778821',)
    
            elif 'restaurant chain' in title or title in ['restaurant']:
                type = ('Q571',)
    
            elif 'city' in title or title in ['location', 'lake', 'sovereign state']:
                type = ('Q2221906',)
    
            elif 'device' in title:
                type = ('Q2858615',)
    
            elif 'room' in title:
                type = ('Q1318740',)
    
            elif title in ['dog breed']:
                type = ('Q144',)
    
            else:
                type = (type,)

    return type


class Bootleg(object):
    
    def __init__(self, args):
        self.args = args
        
        self.model_dir = f'{self.args.database_dir}/{self.args.bootleg_model}'
        self.config_path = f'{self.model_dir}/bootleg_config.yaml'
        self.cand_map = f'{self.args.database_dir}/wiki_entity_data/entity_mappings/alias2qids.json'

        self.entity_dir = f'{self.args.database_dir}/wiki_entity_data'
        self.embed_dir = f'{self.args.database_dir}/wiki_entity_data'
        
        self.almond_domains = args.almond_domains
        self.bootleg_post_process_types = args.bootleg_post_process_types
        
        with open(f'{self.args.database_dir}/wiki_entity_data/type_mappings/wiki/qid2typenames.json', 'r') as fin:
            self.qid2typenames = ujson.load(fin)
        with open(f'{self.args.database_dir}/wiki_entity_data/type_mappings/wiki/type_vocab_to_wikidataqid.json', 'r') as fin:
            self.type_vocab_to_wikidataqid = ujson.load(fin)
            self.wikidataqid_to_type_vocab = {v: k for k, v in self.type_vocab_to_wikidataqid.items()}
        with open(f'{self.args.database_dir}/es_material/type2id.json', 'r') as fin:
            self.type2id = ujson.load(fin)
            
        self.cur_entity_embed_size = 0
        
        # Mapping between model directory and checkpoint name
        model2checkpoint = {'bootleg_uncased_mini': 'bootleg_wiki.pth',
                            'bootleg_uncased_super_mini': 'bootleg_wiki.pth'}
        
        self.ckpt_name, extension = model2checkpoint[self.args.bootleg_model].split('.')
        self.model_ckpt_path = os.path.join(self.model_dir, self.ckpt_name + '.' + extension)
        
        
        self.fixed_overrides = [
            # emmental configs
            "--emmental.dataparallel", 'False',
            "--emmental.log_path", self.args.bootleg_output_dir,
            "--emmental.use_exact_log_path", 'True',
            "--emmental.model_path", self.model_ckpt_path,
            
            # run configs
            "--run_config.dataset_threads", str(getattr(self.args, 'bootleg_dataset_threads', 1)),
            "--run_config.dataloader_threads", str(getattr(self.args, 'bootleg_dataset_threads', 1)),
            "--run_config.eval_batch_size", str(getattr(self.args, 'bootleg_dataset_threads', 32)),
            "--run_config.log_level", 'DEBUG',
            
            # data configs
            "--data_config.print_examples_prep", 'False',
            "--data_config.entity_dir", self.entity_dir,
            "--data_config.entity_prep_dir", "prep",
            "--data_config.emb_dir", self.embed_dir,
            "--data_config.alias_cand_map", 'alias2qids.json',
            "--data_config.word_embedding.cache_dir", self.args.embeddings,
            "--data_config.print_examples", 'False'
        ]
    
    def create_config(self, overrides):
        config_args = parse_boot_and_emm_args(self.config_path, overrides)
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
        logger.info('Extracting mentions...')
        extract_mentions(in_filepath=jsonl_input_path, out_filepath=jsonl_output_path, cand_map_file=self.cand_map,
                         max_alias_len=self.args.max_entity_len,
                         num_workers=getattr(self.args, 'bootleg_extract_num_workers', 32), verbose=False)

    def pad_values(self, tokens, max_size, pad_id):
        if len(tokens) > max_size:
            tokens = tokens[:max_size]
        else:
            tokens += [pad_id] * (max_size - len(tokens))
        return tokens

    def disambiguate_mentions(self, config_args):
        run_model(self.args.bootleg_dump_mode, config_args)

    def collect_features_per_line(self, line, threshold):
        
        tokenized = line['sentence'].split(' ')
        tokens_type_ids = [[self.args.ned_features_default_val[0]] * self.args.ned_features_size[0] for _ in range(len(tokenized))]
        tokens_type_probs = [[self.args.ned_features_default_val[1]] * self.args.ned_features_size[1] for _ in range(len(tokenized))]
    
        for alias, all_qids, all_probs, span in zip(line['aliases'], line['cands'], line['cand_probs'], line['spans']):
            # filter qids with probability lower than a threshold
            idx = reverse_bisect_left(all_probs, threshold)
            all_qids = all_qids[:idx]
            all_probs = all_probs[:idx]

            type_ids = []
            type_probs = []
        
            if not is_banned(alias):
                for qid, prob in zip(all_qids, all_probs):
                    # get all type for a qid
                    all_types = []
                    if qid in self.qid2typenames and self.qid2typenames[qid]:
                        # map entity qid to its type titles on wikidata ; then map titles to their wikidata qids
                        for typename in self.qid2typenames[qid]:
                            if typename in self.type_vocab_to_wikidataqid:
                                all_types.append(self.type_vocab_to_wikidataqid[typename])
                
                    if len(all_types):
                        # update
                        # go through all types
                        for type in all_types:
                            if type in self.type2id:
                                # map wikidata types to thingtalk types
                                if self.bootleg_post_process_types:
                                    # map qid to title
                                    title = self.wikidataqid_to_type_vocab[type]
                                    # process may return multiple types for a single type when it's ambiguous
                                    types = post_process_bootleg_types(qid, type, title, self.almond_domains)
                                    if isinstance(types, str):
                                        types = tuple([types])
                                else:
                                    types = tuple([type])
                                
                                for type_ in types:
                                    type_id = self.type2id[type_]
                                    type_ids.append(type_id)
                                    type_probs.append(prob)
            
                padded_type_ids = self.pad_values(type_ids, self.args.ned_features_size[0], self.args.ned_features_default_val[0])
                padded_type_probs = self.pad_values(type_probs, self.args.ned_features_size[1], self.args.ned_features_default_val[1])
            
                tokens_type_ids[span[0]:span[1]] = [padded_type_ids] * (span[1] - span[0])
                tokens_type_probs[span[0]:span[1]] = [padded_type_probs] * (span[1] - span[0])

        return tokens_type_ids, tokens_type_probs
            
    def collect_features(self, file_name, subsample):
        
        all_tokens_type_ids = []
        all_tokens_type_probs = []
        
        threshold = self.args.bootleg_prob_threshold

        with open(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/{self.ckpt_name}/bootleg_labels.jsonl', 'r') as fin:
            for i, line in enumerate(fin):
                if i >= subsample:
                    break
                line = ujson.loads(line)
                tokens_type_ids, tokens_type_probs = self.collect_features_per_line(line, threshold)
                all_tokens_type_ids.append(tokens_type_ids)
                all_tokens_type_probs.append(tokens_type_probs)

        if os.path.exists(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/{self.ckpt_name}/bootleg_embs.npy'):
            with open(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/{self.ckpt_name}/bootleg_embs.npy', 'rb') as fin:
                emb_data = np.load(fin)
                self.cur_entity_embed_size += emb_data.shape[0]
                
        return all_tokens_type_ids, all_tokens_type_probs
    
    
    def merge_embeds(self, file_list):
        all_emb_data = []
        for file_name in file_list:
            emb_file = f'{self.args.bootleg_output_dir}/{file_name}_bootleg/{self.ckpt_name}/bootleg_embs.npy'
            with open(emb_file, 'rb') as fin:
                emb_data = np.load(fin)
                all_emb_data.append(emb_data)

        all_emb_data = np.concatenate(all_emb_data, axis=0)
        
        # add embeddings for the padding and unknown special tokens
        new_emb = np.concatenate([np.zeros([2, all_emb_data.shape[1]], dtype='float'), all_emb_data], axis=0)
        
        os.makedirs(f'{self.args.bootleg_output_dir}/bootleg/eval/{self.ckpt_name}', exist_ok=True)
        np.save(f'{self.args.bootleg_output_dir}/bootleg/eval/{self.ckpt_name}/ent_embedding.npy', new_emb)
