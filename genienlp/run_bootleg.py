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

from .arguments import save_args, post_parse_general
from .data_utils.bootleg import Bootleg
from .util import set_seed


logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument('--root', default='.', type=str,
                        help='root directory for data, results, embeddings, code, etc.')
    parser.add_argument('--save', required=True, type=str, help='where to save results.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--cache', default='.cache/', type=str, help='where to save cached files')
    
    parser.add_argument('--train_languages', type=str,
                        help='used to specify dataset languages used during training for multilingual tasks'
                             'multiple languages for each task should be concatenated with +')
    parser.add_argument('--eval_languages', type=str,
                        help='used to specify dataset languages used during validation for multilingual tasks'
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
    
    parser.add_argument('--database_type', default='json', choices=['json', 'local-elastic', 'remote-elastic'], help='database to interact with for NER')
    
    parser.add_argument('--min_entity_len', type=int, default=2,
                        help='Minimum length for entities when ngrams database_lookup_method is used ')
    parser.add_argument('--max_entity_len', type=int, default=4,
                        help='Maximum length for entities when ngrams database_lookup_method is used ')
    parser.add_argument('--database_dir', type=str, help='Database folder containing all relevant files (e.g. alias2qids, pretrained models for bootleg)')
    
    parser.add_argument('--bootleg_output_dir', type=str, default='results_temp', help='Path to folder where bootleg prepped files should be saved')
    parser.add_argument('--bootleg_model', type=str, help='Bootleg model to use')
    parser.add_argument('--bootleg_kg_encoder_layer', type=str, default=4, help='Number of kg encoder layers for BootlegBertEncoder model')
    parser.add_argument('--bootleg_dump_mode', choices=['dump_preds', 'dump_embs'], default='dump_preds',
                        help='dump_preds will dump only predictions; dump_embs will dump both prediction and embeddings')
    parser.add_argument('--bootleg_batch_size', type=int, default=30, help='Batch size used for inference using bootleg')
    parser.add_argument('--bootleg_prob_threshold', type=float, default=0.5, help='Probability threshold for accepting a candidate for a mention')
    parser.add_argument('--bootleg_dataset_threads', type=int, default=2, help='Number of threads for parallel processing of dataset in bootleg')
    parser.add_argument('--bootleg_dataloader_threads', type=int, default=4, help='Number of threads for parallel loading of datasets in bootleg')
    parser.add_argument('--bootleg_extract_num_workers', type=int, default=8, help='Number of workers for extracing mentions step of bootleg')
    parser.add_argument('--bootleg_post_process_types', action='store_true', help='Postprocess bootleg types')

    parser.add_argument('--verbose', action='store_true', help='Print detected types for each token')
    parser.add_argument('--almond_domains', nargs='+', default=[],
                        help='Domains used for almond dataset; e.g. music, books, ...')
    parser.add_argument('--ned_features', nargs='+', type=str, default=['type', 'freq'],
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


def bootleg_process_splits(args, split, path, task, bootleg, mode='train'):
    examples = split.examples
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
        bootleg.create_jsonl(path, examples, task.is_contextual())
    
        # extract mentions and mention spans in the sentence and write them to output jsonl files
        bootleg.extract_mentions(path)
    
        # find the right entity candidate for each mention
        bootleg.disambiguate_mentions(config_args)
        
    # extract features for each token in input sentence from bootleg outputs
    all_token_type_ids, all_tokens_type_probs = bootleg.collect_features(input_file_name[:-len('_bootleg.jsonl')])
    
    # override examples features with bootleg features
    if mode != 'dump':
        assert len(examples) == len(all_token_type_ids) == len(all_tokens_type_probs)
        for n, (ex, tokens_type_ids, tokens_type_probs) in enumerate(zip(examples, all_token_type_ids, all_tokens_type_probs)):
            if task.is_contextual():
                for i in range(len(tokens_type_ids)):
                    examples[n].question_feature[i].type_id = tokens_type_ids[i]
                    examples[n].question_feature[i].type_prob = tokens_type_probs[i]
                    examples[n].context_plus_question_feature[i + len(ex.context.split(' '))].type_id = tokens_type_ids[
                        i]
                    examples[n].context_plus_question_feature[i + len(ex.context.split(' '))].type_prob = \
                        tokens_type_probs[i]
            
            else:
                for i in range(len(tokens_type_ids)):
                    examples[n].context_feature[i].type_id = tokens_type_ids[i]
                    examples[n].context_feature[i].type_prob = tokens_type_probs[i]
                    examples[n].context_plus_question_feature[i].type_id = tokens_type_ids[i]
                    examples[n].context_plus_question_feature[i].type_prob = tokens_type_probs[i]
            
            context_plus_question_with_types = task.create_sentence_plus_types_tokens(ex.context_plus_question,
                                                                                      ex.context_plus_question_feature,
                                                                                      args.add_types_to_text)
            examples[n] = ex._replace(context_plus_question_with_types=context_plus_question_with_types)
    
    if args.verbose:
        for ex in examples:
            print()
            print(*[f'token: {token}\ttype: {token_type}' for token, token_type in
                    zip(ex.context_plus_question.split(' '), ex.context_plus_question_feature)], sep='\n')


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
    
    for task in args.train_tasks:
        logger.info(f'Loading {task.name}')
        kwargs = {'test': None, 'validation': None}
        kwargs.update(train_eval_shared_kwargs)
        kwargs['all_dirs'] = args.train_languages
        kwargs['cached_path'] = os.path.join(args.cache, task.name)
        if args.use_curriculum:
            kwargs['curriculum'] = True
        
        logger.info(f'Adding {task.name} to training datasets')
        t0 = time.time()
        splits, paths = task.get_splits(args.data, lower=args.lower, **kwargs)
        t1 = time.time()
        logger.info('Data loading took {} sec'.format(t1 - t0))
        assert not splits.eval and not splits.test
        if args.use_curriculum:
            assert splits.aux
            aux_sets.append(splits.aux)
            logger.info(f'{task.name} has {len(splits.aux)} auxiliary examples')
        else:
            assert splits.train

        bootleg_process_splits(args, splits.train, paths.train, task, bootleg, mode='dump')

        logger.info(f'{task.name} has {len(splits.train)} training examples')
        
        logger.info(f'train all_schema_types: {task.all_schema_types}')
        
        if task.name.startswith('almond'):
            if args.ned_features_default_val:
                args.db_unk_id = int(args.ned_features_default_val[0])
            else:
                args.db_unk_id = 0
            if args.do_ned:
                if getattr(task, 'db', None):
                    args.num_db_types = len(task.db.type2id)
                elif getattr(task, 'bootleg', None):
                    args.num_db_types = len(task.bootleg.type2id)
            else:
                args.num_db_types = 0
            save_args(args, force_overwrite=True)
    
    for task in args.val_tasks:
        logger.info(f'Loading {task.name}')
        kwargs = {'train': None, 'test': None}
        # choose best model based on this dev set
        if args.eval_set_name is not None:
            kwargs['validation'] = args.eval_set_name
        kwargs.update(train_eval_shared_kwargs)
        kwargs['all_dirs'] = args.eval_languages
        kwargs['cached_path'] = os.path.join(args.cache, task.name)
        
        logger.info(f'Adding {task.name} to validation datasets')
        splits, paths = task.get_splits(args.data, lower=args.lower, **kwargs)

        assert not splits.train and not splits.test and not splits.aux
        logger.info(f'{task.name} has {len(splits.eval)} validation examples')

        bootleg_process_splits(args, splits.eval, paths.eval, task, bootleg, mode='dump')

        logger.info(f'eval all_schema_types: {task.all_schema_types}')
        
        # merge bootleg embedding for different splits
        emb_file_list = ['train', args.eval_set_name if args.eval_set_name is not None else 'eval']
        if args.use_curriculum:
            emb_file_list += ['aux']
        bootleg.merge_embeds(emb_file_list)

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
    
    