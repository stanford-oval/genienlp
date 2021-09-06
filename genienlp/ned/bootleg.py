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

import torch
import ujson
from bootleg.end2end.bootleg_annotator import BootlegAnnotator as Annotator
from bootleg.end2end.extract_mentions import extract_mentions
from bootleg.run import run_model
from bootleg.utils.parser.parser_utils import parse_boot_and_emm_args

from ..util import get_devices
from . import AbstractEntityDisambiguator
from .ned_utils import is_banned, reverse_bisect_left

logger = logging.getLogger(__name__)


class BatchBootlegEntityDisambiguator(AbstractEntityDisambiguator):
    '''
    A wrapper for all functionalities needed from bootleg. It takes care of data preprocessing,
    running examples through bootleg, and overriding examples features with the extracted ones
    '''

    def __init__(self, args):
        super().__init__(args)
        logger.info('Initializing Bootleg class')

        ### bootleg specific attribtues
        self.model_dir = f'{self.args.database_dir}/{self.args.bootleg_model}'
        self.config_path = f'{self.model_dir}/bootleg_config.yaml'
        self.cand_map = f'{self.args.database_dir}/wiki_entity_data/entity_mappings/alias2qids.json'

        self.entity_dir = f'{self.args.database_dir}/wiki_entity_data'
        self.embed_dir = f'{self.args.database_dir}/wiki_entity_data'

        with open(f'{self.args.database_dir}/wiki_entity_data/type_mappings/wiki/qid2typenames.json') as fin:
            self.entityqid2typenames = ujson.load(fin)

        self.model_ckpt_path = os.path.join(self.model_dir, 'bootleg_wiki.pth')

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
            "--emmental.device",
            str(getattr(self.args, 'bootleg_device', 0)),
            # run configs
            "--run_config.dataset_threads",
            str(getattr(self.args, 'bootleg_dataset_threads', 1)),
            "--run_config.dataloader_threads",
            str(getattr(self.args, 'bootleg_dataloader_threads', 2)),
            "--run_config.eval_batch_size",
            str(getattr(self.args, 'bootleg_batch_size', 50)),
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
        )

    def disambiguate_mentions(self, config_args):
        run_model(self.args.bootleg_dump_mode, config_args)

    def collect_features_per_alias(self, alias, all_probs, all_qids):
        type_ids = []
        type_probs = []
        qids = []

        # sort candidates based on bootleg's confidence scores (i.e. probabilities)
        # this used to be in bootleg code but was removed in recent version
        packed_list = zip(all_probs, all_qids)
        packed_list_sorted = sorted(packed_list, key=lambda item: item[0], reverse=True)
        all_probs, all_qids = list(zip(*packed_list_sorted))

        # filter qids with probability lower than a threshold
        idx = reverse_bisect_left(all_probs, self.args.bootleg_prob_threshold)
        all_qids = all_qids[:idx]
        all_probs = all_probs[:idx]

        if len(all_qids) > self.args.max_qids_per_entity:
            all_qids = all_qids[: self.args.max_qids_per_entity]
            all_probs = all_probs[: self.args.max_qids_per_entity]

        if is_banned(alias):
            return type_ids, type_probs, qids

        for qid, prob in zip(all_qids, all_probs):
            # to map qids to unique ids we just need to remove the Q character as qids are distinct
            qids.append(int(qid[1:]))

            # get all types for a qid
            all_types = []
            if 'books' in self.args.ned_domains:
                if alias in ['houghton mifflin', 'ciudad de buenos aires']:
                    all_types = ['award']
                elif alias in ['blanche', 'biblioteca breve', '3rd edition', '4th edition']:
                    all_types = ['version, edition, or translation']

            if qid in self.entityqid2typenames and self.entityqid2typenames[qid]:
                # map entity qid to its types on wikidata
                all_types.extend(self.entityqid2typenames[qid])

            if len(all_types) == 0:
                continue

            count = 0
            # go through all types
            for type in all_types:
                if count >= self.args.max_types_per_qid:
                    break

                new_typeqid = None
                if self.args.ned_normalize_types != 'off':
                    new_type = self.normalize_types(type)
                    if new_type is not None:
                        new_typeqid = self.type_vocab_to_typeqid[new_type]

                if new_typeqid is None:
                    if self.args.ned_normalize_types == 'strict':
                        # attempt to normalize type failed; ignore current type
                        continue
                    else:
                        if type not in self.type_vocab_to_typeqid:
                            continue
                        # attempt to normalize type failed; use original type
                        new_typeqid = self.type_vocab_to_typeqid[type]

                if new_typeqid not in self.typeqid2id:
                    continue
                type_id = self.typeqid2id[new_typeqid]
                if type_id in type_ids:
                    continue
                type_ids.append(type_id)
                type_probs.append(prob)
                count += 1

        return type_ids, type_probs, qids

    def collect_features_per_line(self, line):
        tokenized = line['sentence'].split(' ')
        tokens_type_ids = [[0] * self.max_features_size for _ in range(len(tokenized))]
        tokens_type_probs = [[0] * self.max_features_size for _ in range(len(tokenized))]
        tokens_qids = [[0] * self.max_features_size for _ in range(len(tokenized))]

        for alias, all_qids, all_probs, span in zip(line['aliases'], line['cands'], line['cand_probs'], line['spans']):
            type_ids, type_probs, qids = self.collect_features_per_alias(alias, all_probs, all_qids)

            padded_type_ids = self.pad_features(type_ids, self.max_features_size, 0)
            padded_type_probs = self.pad_features(type_probs, self.max_features_size, 0)
            padded_qids = self.pad_features(qids, self.max_features_size, -1)

            tokens_type_ids[span[0] : span[1]] = [padded_type_ids] * (span[1] - span[0])
            tokens_type_probs[span[0] : span[1]] = [padded_type_probs] * (span[1] - span[0])
            tokens_qids[span[0] : span[1]] = [padded_qids] * (span[1] - span[0])

        return tokens_type_ids, tokens_type_probs, tokens_qids

    def process_examples(self, examples, split_path, utterance_field):
        # extract features for each token in input sentence from bootleg outputs
        all_token_type_ids, all_token_type_probs, all_token_qids = [], [], []

        file_name = os.path.basename(split_path.rsplit('.', 1)[0])

        with open(f'{self.args.bootleg_output_dir}/{file_name}_bootleg/bootleg_wiki/bootleg_labels.jsonl', 'r') as fin:
            for i, line in enumerate(fin):
                if i >= self.args.subsample:
                    break
                line = ujson.loads(line)
                tokens_type_ids, tokens_type_probs, tokens_qids = self.collect_features_per_line(line)
                all_token_type_ids.append(tokens_type_ids)
                all_token_type_probs.append(tokens_type_probs)
                all_token_qids.append(tokens_qids)

        all_token_type_ids = all_token_type_ids[: self.args.subsample]
        all_token_type_probs = all_token_type_probs[: self.args.subsample]
        all_token_qids = all_token_qids[: self.args.subsample]

        self.replace_features_inplace(examples, all_token_type_ids, all_token_type_probs, all_token_qids, utterance_field)

    def dump_entities_with_labels(self, examples, path, utterance_field):
        input_file_dir = os.path.dirname(path)
        input_file_name = os.path.basename(path.rsplit('.', 1)[0] + '_bootleg.jsonl')
        data_overrides = ["--data_config.data_dir", input_file_dir, "--data_config.test_dataset.file", input_file_name]

        # get config args
        config_overrides = self.fixed_overrides
        config_overrides.extend(data_overrides)
        config_args = self.create_config(config_overrides)

        # create jsonl files from input examples
        # jsonl is the input format bootleg expects
        self.create_jsonl(path, examples, utterance_field)

        # extract mentions and mention spans in the sentence and write them to output jsonl files
        self.extract_mentions(path)

        # find the right entity candidate for each mention
        self.disambiguate_mentions(config_args)


class ServingBootlegEntityDisambiguator(BatchBootlegEntityDisambiguator):
    '''
    BootlegAnnotator is a wrapper for bootleg's native annotator which takes care of bootleg instantiations and
    extracting required features from examples on-the-fly
    '''

    def __init__(self, args):
        super().__init__(args)
        bootleg_config = self.create_config(self.fixed_overrides)
        device = get_devices()[0]  # server only runs on a single device

        # instantiate the annotator class. we use annotator only in server mode.
        # for training we use bootleg functions which preprocess and cache data using multiprocessing, and batching to speed up NED
        self.annotator = Annotator(
            config=bootleg_config,
            device='cpu' if device.type == 'cpu' else 'cuda',
            min_alias_len=args.min_entity_len,
            max_alias_len=args.max_entity_len,
            cand_map=self.cand_map,
            threshold=args.bootleg_prob_threshold,
            model_name=args.bootleg_model,
            verbose=False,
        )
        # collect all outputs now; we will filter later
        self.annotator.set_threshold(0.0)

    def process_examples(self, examples, split_path, utterance_field):
        with torch.no_grad():
            bootleg_inputs = []
            for ex in examples:
                bootleg_inputs.append(getattr(ex, utterance_field))

            bootleg_labels = self.annotator.label_mentions(bootleg_inputs)

            keys = tuple(bootleg_labels.keys())
            values = list(bootleg_labels.values())
            values_unpacked = list(zip(*values))

            bootleg_labels_unpacked = [dict(zip(keys, values)) for values in values_unpacked]

            all_token_type_ids, all_token_type_probs, all_token_qids = [], [], []
            for ex, label in zip(examples, bootleg_labels_unpacked):
                line = {}
                line['sentence'] = getattr(ex, utterance_field)

                assert len(label) == 7
                line['aliases'], line['spans'], line['cands'] = label['aliases'], label['spans'], label['cands']
                line['cand_probs'] = list(map(lambda item: list(item), label['cand_probs']))

                tokens_type_ids, tokens_type_probs, tokens_qids = self.collect_features_per_line(line)
                all_token_type_ids.append(tokens_type_ids)
                all_token_type_probs.append(tokens_type_probs)
                all_token_qids.append(tokens_qids)

            self.replace_features_inplace(examples, all_token_type_ids, all_token_type_probs, all_token_qids, utterance_field)
