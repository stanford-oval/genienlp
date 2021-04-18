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
import logging
import time
from pprint import pformat
import shutil

import ujson

from .arguments import save_args, post_parse_general
from .data_utils.bootleg import Bootleg
from .util import set_seed


logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument('--root', default='.', type=str,
                        help='root directory for data, results, embeddings, code, etc.')
    parser.add_argument('--save', required=True, type=str, help='where to save results.')
    parser.add_argument('--embeddings', default='.embeddings/', type=str, help='where to save embeddings.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--cache', default='.cache/', type=str, help='where to save cached files')

    parser.add_argument('--train_languages', type=str, default='en', dest='train_src_languages',
                        help='Specify dataset source languages used during training for multilingual tasks'
                             'multiple languages for each task should be concatenated with +')
    parser.add_argument('--eval_languages', type=str, default='en', dest='eval_src_languages',
                        help='Specify dataset source languages used during validation for multilingual tasks'
                             'multiple languages for each task should be concatenated with +')

    parser.add_argument('--train_tgt_languages', type=str, default='en',
                        help='Specify dataset target languages used during training for multilingual tasks'
                             'multiple languages for each task should be concatenated with +')
    parser.add_argument('--eval_tgt_languages', type=str, default='en',
                        help='Specify dataset target languages used during validation for multilingual tasks'
                             'multiple languages for each task should be concatenated with +')

    parser.add_argument('--train_tasks', nargs='+', type=str, dest='train_task_names', help='tasks to use for training',
                        required=True)
    
    parser.add_argument('--val_tasks', nargs='+', type=str, dest='val_task_names',
                        help='tasks to collect evaluation metrics for')

    parser.add_argument('--val_batch_size', nargs='+', default=[256], type=int,
                        help='Batch size for validation corresponding to tasks in val tasks')

    parser.add_argument('--sentence_batching', action='store_true',
                        help='Batch same sentences together (used for multilingual tasks)')
    parser.add_argument('--train_batch_size', type=int, default=0,
                        help='Number of samples to use in each batch; will be used instead of train_batch_tokens when sentence_batching is on')

    parser.add_argument('--eval_set_name', type=str, help='Evaluation dataset name to use during training')

    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the datasets')
    parser.add_argument('--preserve_case', action='store_false', dest='lower',
                        help='whether to preserve casing for all text')

    parser.add_argument("--almond_has_single_program", action='store_false', dest='almond_has_multiple_programs', help='Indicate if almond dataset has multiple programs for each sentence')

    parser.add_argument('--num_workers', type=int, default=0, help='Number of processes to use for data loading (0 means no multiprocessing)')
    
    parser.add_argument('--database_type', default='json', choices=['json', 'remote-elastic'], help='database to interact with for NER')
    
    parser.add_argument('--min_entity_len', type=int, default=1,
                        help='Minimum length for entities when ngrams database_lookup_method is used ')
    parser.add_argument('--max_entity_len', type=int, default=4,
                        help='Maximum length for entities when ngrams database_lookup_method is used ')
    parser.add_argument('--database_dir', type=str, default='database/', help='Database folder containing all relevant files (e.g. alias2qids, pretrained models for bootleg)')
    
    parser.add_argument('--bootleg_output_dir', type=str, default='results_temp', help='Path to folder where bootleg prepped files should be saved')
    parser.add_argument('--bootleg_model', type=str, default='bootleg_uncased_mini', help='Bootleg model to use')
    parser.add_argument('--bootleg_dump_mode', choices=['dump_preds', 'dump_embs'], default='dump_preds',
                        help='dump_preds will dump only predictions; dump_embs will dump both prediction and embeddings')
    parser.add_argument('--bootleg_batch_size', type=int, default=32, help='Batch size used for inference using bootleg')
    parser.add_argument('--bootleg_prob_threshold', type=float, default=0.3, help='Probability threshold for accepting a candidate for a mention')
    parser.add_argument('--bootleg_dataset_threads', type=int, default=1, help='Number of threads for parallel processing of dataset in bootleg')
    parser.add_argument('--bootleg_dataloader_threads', type=int, default=1, help='Number of threads for parallel loading of datasets in bootleg')
    parser.add_argument('--bootleg_extract_num_workers', type=int, default=32, help='Number of workers for extracing mentions step of bootleg')
    parser.add_argument('--bootleg_post_process_types', action='store_true', help='Postprocess bootleg types')

    parser.add_argument('--verbose', action='store_true', help='Print detected types for each token')
    parser.add_argument('--almond_domains', nargs='+', default=[],
                        help='Domains used for almond dataset; e.g. music, books, ...')
    parser.add_argument('--ned_features', nargs='+', type=str, default=['type_id', 'type_prob'],
                        help='Features that will be extracted for each entity: "type" and "freq" are supported.'
                             ' Order is important')
    parser.add_argument('--ned_features_size', nargs='+', type=int, default=[1, 1],
                        help='Max length of each feature vector. All features are padded up to this length')
    parser.add_argument('--ned_features_default_val', nargs='+', type=float, default=[0, 1.0],
                        help='Max length of each feature vector. All features are padded up to this length')
    
    parser.add_argument('--almond_lang_as_question', action='store_true',
                        help='if true will use "Translate from ${language} to ThingTalk" for question')
    parser.add_argument('--almond_detokenize_sentence', action='store_true',
                        help='undo word tokenization of almond sentence fields (useful if the tokenizer is sentencepiece)')
    parser.add_argument('--almond_thingtalk_version', type=int, choices=[1, 2], default=2, help='Thingtalk version for almond datasets')

    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used for training')
    
    parser.add_argument('--no_commit', action='store_false', dest='commit',
                        help='do not track the git commit associated with this training run')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Ok if the save directory already exists, i.e. overwrite is ok')
    
    parser.add_argument('--skip_cache', action='store_true',
                        help='whether to use exisiting cached splits or generate new ones')
    parser.add_argument('--cache_input_data', action='store_true',
                        help='Cache examples from input data for faster subsequent trainings')
    
    parser.add_argument('--use_curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--aux_dataset', default='', type=str,
                        help='path to auxiliary dataset (ignored if curriculum is not used)')
    parser.add_argument("--add_types_to_text", default='no', choices=['no', 'insert', 'append'])

    # token classification task args
    parser.add_argument('--num_labels', type=int, help='num_labels for classification tasks')
    parser.add_argument('--ner_domains', nargs='+', type=str, help='domains to use for CrossNER task')
    parser.add_argument('--hf_test_overfit', action='store_true', help='Debugging flag for hf datasets where validation will be performed on train set')


def bootleg_process_splits(args, examples, path, task, bootleg, mode='train'):
    config_overrides = bootleg.fixed_overrides
    
    input_file_dir = os.path.dirname(path)
    input_file_name = os.path.basename(path.rsplit('.', 1)[0] + '_bootleg.jsonl')
    
    data_overrides = [
        "--data_config.data_dir", input_file_dir,
        "--data_config.test_dataset.file", input_file_name
    ]
    
    # get config args
    config_overrides.extend(data_overrides)
    config_args = bootleg.create_config(config_overrides)
    
    if mode == 'dump':
        # create jsonl files from input examples
        # jsonl is the input format bootleg expects
        bootleg.create_jsonl(path, examples, task.utterance_field())
    
        # extract mentions and mention spans in the sentence and write them to output jsonl files
        bootleg.extract_mentions(path)
    
        # find the right entity candidate for each mention
        bootleg.disambiguate_mentions(config_args)
        
    # extract features for each token in input sentence from bootleg outputs
    all_token_type_ids, all_tokens_type_probs = bootleg.collect_features(input_file_name[:-len('_bootleg.jsonl')],
                                                                         args.subsample,
                                                                         getattr(task, 'TTtype2qid', None))
    
    all_token_type_ids = all_token_type_ids[:args.subsample]
    all_tokens_type_probs = all_tokens_type_probs[:args.subsample]
    
    # override examples features with bootleg features
    if mode != 'dump':
        assert len(examples) == len(all_token_type_ids) == len(all_tokens_type_probs)
        for n, (ex, tokens_type_ids, tokens_type_probs) in enumerate(zip(examples, all_token_type_ids, all_tokens_type_probs)):
            if task.utterance_field() == 'question':
                for i in range(len(tokens_type_ids)):
                    examples[n].question_feature[i].type_id = tokens_type_ids[i]
                    examples[n].question_feature[i].type_prob = tokens_type_probs[i]
                question_plus_types = task.add_type_tokens(ex.question, ex.question_feature, args.add_types_to_text)
                examples[n].question_plus_types = question_plus_types

            else:
                # context is the utterance field
                for i in range(len(tokens_type_ids)):
                    examples[n].context_feature[i].type_id = tokens_type_ids[i]
                    examples[n].context_feature[i].type_prob = tokens_type_probs[i]
                context_plus_types = task.add_type_tokens(ex.context, ex.context_feature, args.add_types_to_text)
                examples[n].context_plus_types = context_plus_types

    if args.verbose:
        for ex in examples:
            print()
            print(*[f'context token: {token}\ttype: {token_type}' for token, token_type in
                    zip(ex.context.split(' '), ex.context_plus_feature)], sep='\n')
            print(*[f'question token: {token}\ttype: {token_type}' for token, token_type in
                    zip(ex.question.split(' '), ex.question_plus_feature)], sep='\n')


def dump_bootleg_features(args, logger):

    bootleg = Bootleg(args)
    
    train_sets, val_sets, aux_sets = [], [], []
    
    train_eval_shared_kwargs = {'subsample': args.subsample,
                                'skip_cache': args.skip_cache,
                                'cache_input_data': args.cache_input_data,
                                'sentence_batching': args.sentence_batching,
                                'almond_lang_as_question': args.almond_lang_as_question,
                                'num_workers': args.num_workers,
                                }
    
    assert len(args.train_tasks) == len(args.val_tasks)
    
    for train_task, val_task in zip(args.train_tasks, args.val_tasks):
        
        # process train split
        logger.info(f'Loading {train_task.name}')
        kwargs = {'test': None, 'validation': None}
        kwargs.update(train_eval_shared_kwargs)
        kwargs['all_dirs'] = args.train_src_languages
        kwargs['cached_path'] = os.path.join(args.cache, train_task.name)
        kwargs['ner_domains'] = args.ner_domains
        if args.use_curriculum:
            kwargs['curriculum'] = True
    
        logger.info(f'Adding {train_task.name} to training datasets')
        t0 = time.time()
        splits, paths = train_task.get_splits(args.data, lower=args.lower, **kwargs)
        t1 = time.time()
        logger.info('Data loading took {} sec'.format(t1 - t0))
        assert not splits.eval and not splits.test
        if args.use_curriculum:
            assert splits.aux
            aux_sets.append(splits.aux)
            logger.info(f'{train_task.name} has {len(splits.aux)} auxiliary examples')
        else:
            assert splits.train
            
        train_dataset = splits.train
        train_path = paths.train
        
        logger.info(f'{train_task.name} has {len(splits.train)} training examples')

        if hasattr(train_task, 'all_schema_types'):
            logger.info(f'train all_schema_types: {train_task.all_schema_types}')
    
        if train_task.name.startswith('almond'):
            if args.ned_features_default_val:
                args.db_unk_id = int(args.ned_features_default_val[0])
            else:
                args.db_unk_id = 0
            if args.do_ned:
                if bootleg:
                    args.num_db_types = len(bootleg.typeqid2id)
                elif getattr(train_task, 'db', None):
                    args.num_db_types = len(train_task.db.typeqid2id)
            else:
                args.num_db_types = 0
        else:
            args.db_unk_id = 0
            args.num_db_types = 0
        save_args(args, force_overwrite=True)

        # process validation split
        logger.info(f'Loading {val_task.name}')
        kwargs = {'train': None, 'test': None}
        # choose best model based on this dev set
        if args.eval_set_name is not None:
            kwargs['validation'] = args.eval_set_name
        kwargs.update(train_eval_shared_kwargs)
        kwargs['all_dirs'] = args.eval_src_languages
        kwargs['cached_path'] = os.path.join(args.cache, val_task.name)
        kwargs['ner_domains'] = args.ner_domains
        kwargs['hf_test_overfit'] = args.hf_test_overfit

        logger.info(f'Adding {val_task.name} to validation datasets')
        splits, paths = val_task.get_splits(args.data, lower=args.lower, **kwargs)

        assert not splits.train and not splits.test and not splits.aux
        logger.info(f'{val_task.name} has {len(splits.eval)} validation examples')

        if hasattr(val_task, 'all_schema_types'):
            logger.info(f'eval all_schema_types: {val_task.all_schema_types}')
        
        eval_dataset = splits.eval
        
        # merge all splits before feeding to bootleg
        all_examples = train_dataset.examples + eval_dataset.examples
        dir_name = os.path.dirname(train_path)
        extension = train_path.rsplit('.', 1)[1]
        all_paths = os.path.join(dir_name, 'combined' + '.' + extension)
        
        assert train_task == val_task
        bootleg_process_splits(args, all_examples, all_paths, train_task, bootleg, mode='dump')
        
        eval_file_name = args.eval_set_name if args.eval_set_name is not None else 'eval'

        train_output_path = f'{args.bootleg_output_dir}/train_bootleg/{bootleg.ckpt_name}'
        eval_output_path = f'{args.bootleg_output_dir}/{eval_file_name}_bootleg/{bootleg.ckpt_name}'
        os.makedirs(train_output_path, exist_ok=True)
        os.makedirs(eval_output_path, exist_ok=True)
        train_output_file = open(os.path.join(train_output_path, 'bootleg_labels.jsonl'), 'w')
        eval_output_file = open(os.path.join(eval_output_path, 'bootleg_labels.jsonl'), 'w')
        
        train_size = len(train_dataset.examples)
        eval_size = len(eval_dataset.examples)
        
        # unmerge bootleg dumped labels
        with open(f'{args.bootleg_output_dir}/combined_bootleg/{bootleg.ckpt_name}/bootleg_labels.jsonl', 'r') as fin:
            
            # sort output lines first to align with input (required for bootleg >=1.0.0)
            all_lines = fin.readlines()
            all_sent_ids = [ujson.loads(line)['sent_idx_unq'] for line in all_lines]
            all_lines = list(zip(*sorted(zip(all_sent_ids, all_lines), key=lambda item: item[0])))[1]
            
            i = 0
            for line in all_lines:
                if i < train_size:
                    train_output_file.write(line)
                else:
                    eval_output_file.write(line)
                i += 1
        
        assert i == train_size + eval_size
        
        # close output files
        train_output_file.close()
        eval_output_file.close()
        
        shutil.rmtree(f'{args.bootleg_output_dir}/combined_bootleg')


    logger.info('Created bootleg features for provided datasets with subsampling: {}'.format(args.subsample))


def main(args):
    
    args.do_ned = True
    args.ned_retrieve_method = 'bootleg'
    args.override_context = None
    args.override_question = None

    args = post_parse_general(args)
    set_seed(args)
    
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    dump_bootleg_features(args, logger)
    
    