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
import shutil
import time
from pprint import pformat

import ujson

from .arguments import post_parse_general
from .ned.bootleg import BatchBootlegEntityDisambiguator
from .util import set_seed

logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument('--root', default='.', type=str, help='root directory for data, results, embeddings, code, etc.')
    parser.add_argument('--save', required=True, type=str, help='where to save results.')
    parser.add_argument('--embeddings', default='.embeddings/', type=str, help='where to save embeddings.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--cache', default='.cache/', type=str, help='where to save cached files')

    parser.add_argument(
        '--train_languages',
        type=str,
        default='en',
        dest='train_src_languages',
        help='Specify dataset source languages used during training for multilingual tasks'
        'multiple languages for each task should be concatenated with +',
    )
    parser.add_argument(
        '--eval_languages',
        type=str,
        default='en',
        dest='eval_src_languages',
        help='Specify dataset source languages used during validation for multilingual tasks'
        'multiple languages for each task should be concatenated with +',
    )

    # we set dest to be train_task_names so we can reuse previous code for arguments processing
    parser.add_argument(
        '--tasks',
        nargs='+',
        type=str,
        dest='train_task_names',
        help='tasks to use for bootleg',
        required=True,
    )

    parser.add_argument(
        '--sentence_batching', action='store_true', help='Batch same sentences together (used for multilingual tasks)'
    )

    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the datasets')
    parser.add_argument('--preserve_case', action='store_false', dest='lower', help='whether to preserve casing for all text')

    parser.add_argument(
        "--almond_has_single_program",
        action='store_false',
        dest='almond_has_multiple_programs',
        help='Indicate if almond dataset has multiple programs for each sentence',
    )

    parser.add_argument(
        '--num_workers', type=int, default=0, help='Number of processes to use for data loading (0 means no multiprocessing)'
    )

    parser.add_argument('--min_entity_len', type=int, default=1, help='Minimum token-length of entities retrieved in bootleg')
    parser.add_argument('--max_entity_len', type=int, default=4, help='Maximum token-length of entities retrieved in bootleg')
    parser.add_argument(
        '--database_dir',
        type=str,
        default='database/',
        help='Database folder containing all relevant files (e.g. alias2qids, pretrained models for bootleg)',
    )

    parser.add_argument(
        '--bootleg_output_dir',
        type=str,
        default='results_temp',
        help='Path to folder where bootleg prepped files should be saved',
    )
    parser.add_argument('--bootleg_model', type=str, default='bootleg_uncased_mini', help='Bootleg model to use')
    parser.add_argument(
        '--bootleg_dump_mode',
        choices=['dump_preds', 'dump_embs'],
        default='dump_preds',
        help='dump_preds will dump only predictions; dump_embs will dump both prediction and embeddings',
    )

    parser.add_argument(
        '--bootleg_device', type=int, default=0, help="device to run bootleg predictions on (-1 for cpu or gpu id)"
    )

    parser.add_argument('--bootleg_batch_size', type=int, default=50, help='Batch size used for inference using bootleg')
    parser.add_argument(
        '--bootleg_prob_threshold',
        type=float,
        default=0.3,
        help='Probability threshold for accepting a candidate for a mention',
    )
    parser.add_argument(
        '--bootleg_dataset_threads',
        type=int,
        default=1,
        help='Number of threads for parallel processing of dataset in bootleg',
    )
    parser.add_argument(
        '--bootleg_dataloader_threads',
        type=int,
        default=2,
        help='Number of threads for parallel loading of datasets in bootleg',
    )
    parser.add_argument(
        '--bootleg_extract_num_workers', type=int, default=32, help='Number of workers for extracing mentions step of bootleg'
    )

    parser.add_argument('--ned_domains', nargs='+', default=[], help='Domains used for almond dataset; e.g. music, books, ...')
    parser.add_argument(
        '--bootleg_data_splits',
        nargs='+',
        type=str,
        default=['train', 'eval'],
        help='Data splits to prepare bootleg features for. train and eval should be included by default; test set is optional',
    )

    parser.add_argument(
        '--almond_lang_as_question',
        action='store_true',
        help='if true will use "Translate from ${language} to ThingTalk" for question',
    )
    parser.add_argument(
        '--almond_detokenize_sentence',
        action='store_true',
        help='undo word tokenization of almond sentence fields (useful if the tokenizer is sentencepiece)',
    )
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used for training')

    parser.add_argument(
        '--no_commit',
        action='store_false',
        dest='commit',
        help='do not track the git commit associated with this training run',
    )
    parser.add_argument(
        '--exist_ok', action='store_true', help='Ok if the save directory already exists, i.e. overwrite is ok'
    )

    parser.add_argument('--skip_cache', action='store_true', help='whether to use existing cached splits or generate new ones')
    parser.add_argument(
        '--cache_input_data', action='store_true', help='Cache examples from input data for faster subsequent trainings'
    )

    # token classification task args
    parser.add_argument('--num_labels', type=int, help='num_labels for classification tasks')
    parser.add_argument('--crossner_domains', nargs='+', type=str, help='domains to use for CrossNER task')
    parser.add_argument(
        '--hf_test_overfit',
        action='store_true',
        help='Debugging flag for hf datasets where validation will be performed on train set',
    )


def bootleg_dump_entities(args, logger):
    bootleg = BatchBootlegEntityDisambiguator(args)

    bootleg_shared_kwargs = {
        'subsample': args.subsample,
        'skip_cache': args.skip_cache,
        'cache_input_data': args.cache_input_data,
        'num_workers': args.num_workers,
        'all_dirs': args.train_src_languages,
        'crossner_domains': args.crossner_domains,
    }

    # run_bootleg does not need special treatment for train vs eval/ test
    for task in args.train_tasks:

        task_all_examples = []
        task_all_paths = []

        kwargs = {'train': None, 'validation': None, 'test': None}
        kwargs.update(bootleg_shared_kwargs)
        kwargs['cached_path'] = os.path.join(args.cache, task.name)
        for split in args.bootleg_data_splits:
            if split == 'train':
                del kwargs['train']  # deleting keys means use the default file name
            elif split == 'test':
                del kwargs['test']
            else:
                kwargs['validation'] = split

            logger.info(f'Adding {task.name} to bootleg datasets')
            t0 = time.time()
            splits, paths = task.get_splits(args.data, lower=args.lower, **kwargs)
            t1 = time.time()
            logger.info('Data loading took {:.2f} seconds'.format(t1 - t0))

            split_examples = (
                getattr(splits, split).examples if split in ['train', 'test'] else getattr(splits, 'eval').examples
            )
            split_path = getattr(paths, split) if split in ['train', 'test'] else getattr(paths, 'eval')

            task_all_examples.append(split_examples)
            task_all_paths.append(split_path)

            logger.info(f'{task.name} has {len(split_examples)} examples')

        # merge all splits before feeding to bootleg
        all_examples = [item for examples in task_all_examples for item in examples]
        dir_name = os.path.dirname(task_all_paths[0])
        extension = task_all_paths[0].rsplit('.', 1)[1]
        all_paths = os.path.join(dir_name, 'combined' + '.' + extension)

        bootleg.dump_entities_with_labels(all_examples, all_paths, task.utterance_field)

        # unmerge bootleg dumped labels
        line_number = 0
        with open(f'{args.bootleg_output_dir}/combined_bootleg/bootleg_wiki/bootleg_labels.jsonl', 'r') as fin:

            # sort output lines first to align with input (required for bootleg >=1.0.0)
            all_lines = fin.readlines()
            all_sent_ids = [ujson.loads(line)['sent_idx_unq'] for line in all_lines]
            all_lines = list(zip(*sorted(zip(all_sent_ids, all_lines), key=lambda item: item[0])))[1]

            for i, split in enumerate(args.bootleg_data_splits):
                output_path = f'{args.bootleg_output_dir}/{split}_bootleg/bootleg_wiki'
                os.makedirs(output_path, exist_ok=True)
                output_file = open(os.path.join(output_path, 'bootleg_labels.jsonl'), 'w')
                split_size = len(task_all_examples[i])

                for _ in range(split_size):
                    output_file.write(all_lines[line_number])
                    line_number += 1

                output_file.close()

            assert line_number == sum(map(lambda examples: len(examples), task_all_examples))

        shutil.rmtree(f'{args.bootleg_output_dir}/combined_bootleg')

    logger.info('Created bootleg features for provided datasets with subsampling: {}'.format(args.subsample))


def main(args):

    args.do_ned = True
    args.ned_retrieve_method = 'bootleg'
    args.override_context = None
    args.override_question = None
    args.almond_type_mapping_path = None

    # set these so we can use post_parse_general for train and run_bootleg
    args.val_task_names = None
    args.max_types_per_qid = 0
    args.max_qids_per_entity = 0

    args = post_parse_general(args)
    set_seed(args)

    logger.info(f'Arguments:\n{pformat(vars(args))}')

    bootleg_dump_entities(args, logger)
