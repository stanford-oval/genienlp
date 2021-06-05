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

import logging
import os

import numpy as np
import torch
import ujson
from bootleg.end2end.bootleg_annotator import BootlegAnnotator
from bootleg.end2end.extract_mentions import extract_mentions
from bootleg.run import run_model
from bootleg.utils.parser.parser_utils import parse_boot_and_emm_args

from .database_utils import is_banned

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
    tokens_type_ids, tokens_type_probs = bootleg_annotator.bootleg.collect_features_per_line(
        line, args.bootleg_prob_threshold, getattr(task, 'TTtype2qid', None)
    )

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
        min_alias_len=args.min_entity_len,
        max_alias_len=args.max_entity_len,
        cand_map=bootleg.cand_map,
        threshold=args.bootleg_prob_threshold,
        model_name=args.bootleg_model,
        verbose=False,
        neural_ner_model=args.bootleg_neural_ner_model,
        neural_batch_size=args.bootleg_neural_ner_batch_size,
        neural_embeddings_dir=args.embeddings,
    )
    # collect all outputs now; we will filter later
    bootleg_annotator.set_threshold(0.0)
    setattr(bootleg_annotator, 'bootleg', bootleg)
    return bootleg_annotator


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
        with open(f'{self.args.database_dir}/es_material/typeqid2id.json', 'r') as fin:
            self.typeqid2id = ujson.load(fin)

        # almond specific
        with open(f'{self.args.database_dir}/es_material/almond_type_mapping_matching.json', 'r') as fin:
            self.almond_type_mapping_match = ujson.load(fin)
        with open(f'{self.args.database_dir}/es_material/almond_type_mapping_inclusion.json', 'r') as fin:
            self.almond_type_mapping_include = ujson.load(fin)

        self.cur_entity_embed_size = 0

        # Mapping between model directory and checkpoint name
        model2checkpoint = {'bootleg_uncased_mini': 'bootleg_wiki.pth', 'bootleg_uncased_super_mini': 'bootleg_wiki.pth'}

        self.ckpt_name, extension = model2checkpoint[self.args.bootleg_model].split('.')
        self.model_ckpt_path = os.path.join(self.model_dir, self.ckpt_name + '.' + extension)

        self.fixed_overrides = [
            # emmental configs
            "--emmental.dataparallel",
            'False',
            "--emmental.log_path",
            self.args.bootleg_output_dir,
            "--emmental.use_exact_log_path",
            'True',
            "--emmental.model_path",
            self.model_ckpt_path,
            # run configs
            "--run_config.dataset_threads",
            str(getattr(self.args, 'bootleg_dataset_threads', 1)),
            "--run_config.dataloader_threads",
            str(getattr(self.args, 'bootleg_dataloader_threads', 1)),
            "--run_config.eval_batch_size",
            str(getattr(self.args, 'bootleg_batch_size', 32)),
            "--run_config.log_level",
            'DEBUG',
            # data configs
            "--data_config.print_examples_prep",
            'False',
            "--data_config.entity_dir",
            self.entity_dir,
            "--data_config.entity_prep_dir",
            "prep",
            "--data_config.emb_dir",
            self.embed_dir,
            "--data_config.alias_cand_map",
            'alias2qids.json',
            "--data_config.word_embedding.cache_dir",
            self.args.embeddings,
            "--data_config.print_examples",
            'False',
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
        extract_mentions(
            in_filepath=jsonl_input_path,
            out_filepath=jsonl_output_path,
            cand_map_file=self.cand_map,
            min_alias_len=self.args.min_entity_len,
            max_alias_len=self.args.max_entity_len,
            num_workers=getattr(self.args, 'bootleg_extract_num_workers', 32),
            verbose=False,
            neural_ner_model=self.args.bootleg_neural_ner_model,
            neural_batch_size=self.args.bootleg_neural_ner_batch_size,
            neural_embeddings_dir=self.args.embeddings,
        )

    def pad_values(self, tokens, max_size, pad_id):
        if len(tokens) > max_size:
            tokens = tokens[:max_size]
        else:
            tokens += [pad_id] * (max_size - len(tokens))
        return tokens

    def disambiguate_mentions(self, config_args):
        run_model(self.args.bootleg_dump_mode, config_args)

    def post_process_bootleg_types(self, title, TTtype2qid):
        type_mapping_match = dict()
        type_mapping_include = dict()
        for domain in self.almond_domains:
            type_mapping_match.update(self.almond_type_mapping_match[domain])
            type_mapping_include.update(self.almond_type_mapping_match[domain])

        types = None
        if title in type_mapping_match:
            types = type_mapping_match[title]
        else:
            for key in type_mapping_include.keys():
                if key in title:
                    types = type_mapping_include[key]

        typeqids = None
        if types is not None:
            if isinstance(types, str):
                typeqids = [TTtype2qid[types]]
            elif isinstance(types, (list, tuple)):
                typeqids = [TTtype2qid[type_] for type_ in types]

        return typeqids

    def collect_features_per_line(self, line, threshold, TTtype2qid):

        tokenized = line['sentence'].split(' ')
        tokens_type_ids = [
            [self.args.ned_features_default_val[0]] * self.args.ned_features_size[0] for _ in range(len(tokenized))
        ]
        tokens_type_probs = [
            [self.args.ned_features_default_val[1]] * self.args.ned_features_size[1] for _ in range(len(tokenized))
        ]

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
                        for typeqid in all_types:
                            if typeqid in self.typeqid2id:
                                # map wikidata types to thingtalk types
                                if self.bootleg_post_process_types:
                                    # map qid to title
                                    title = self.wikidataqid_to_type_vocab[typeqid]
                                    # process may return multiple types for a single type when it's ambiguous
                                    typeqids = self.post_process_bootleg_types(title, TTtype2qid)
                                    if typeqids is None:
                                        typeqids = [typeqid]

                                else:
                                    typeqids = [typeqid]

                                for typeqid_ in typeqids:
                                    type_id = self.typeqid2id[typeqid_]
                                    type_ids.append(type_id)
                                    type_probs.append(prob)

                padded_type_ids = self.pad_values(
                    type_ids, self.args.ned_features_size[0], self.args.ned_features_default_val[0]
                )
                padded_type_probs = self.pad_values(
                    type_probs, self.args.ned_features_size[1], self.args.ned_features_default_val[1]
                )

                tokens_type_ids[span[0] : span[1]] = [padded_type_ids] * (span[1] - span[0])
                tokens_type_probs[span[0] : span[1]] = [padded_type_probs] * (span[1] - span[0])

        return tokens_type_ids, tokens_type_probs

    def collect_features(self, file_name, subsample, TTtype2qid):

        all_tokens_type_ids = []
        all_tokens_type_probs = []

        threshold = self.args.bootleg_prob_threshold

        with open(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/{self.ckpt_name}/bootleg_labels.jsonl', 'r') as fin:
            for i, line in enumerate(fin):
                if i >= subsample:
                    break
                line = ujson.loads(line)
                tokens_type_ids, tokens_type_probs = self.collect_features_per_line(line, threshold, TTtype2qid)
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
