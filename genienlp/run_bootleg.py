import os
import logging
import time

from .arguments import save_args, post_parse_general
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
    parser.add_argument('--train_batch_tokens', nargs='+', default=[9000], type=int,
                        help='Number of tokens to use for dynamic batching, corresponding to tasks in train tasks')

    parser.add_argument('--val_tasks', nargs='+', type=str, dest='val_task_names',
                        help='tasks to collect evaluation metrics for')

    parser.add_argument('--val_batch_size', nargs='+', default=[256], type=int,
                        help='Batch size for validation corresponding to tasks in val tasks')

    parser.add_argument('--paired', action='store_true',
                        help='Pair related examples before numericalizing the input (e.g. training with synthetic and paraphrase '
                             'sentence pairs for almond task)')
    parser.add_argument('--max_pairs', type=int, default=1000000,
                        help='Maximum number of pairs to make for each example group')

    parser.add_argument('--sentence_batching', action='store_true',
                        help='Batch same sentences together (used for multilingual tasks)')
    parser.add_argument('--train_batch_size', type=int, default=0,
                        help='Number of samples to use in each batch; will be used instead of train_batch_tokens when sentence_batching is on')

    parser.add_argument('--eval_set_name', type=str, help='Evaluation dataset name to use during training')

    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the datasets')
    parser.add_argument('--preserve_case', action='store_false', dest='lower',
                        help='whether to preserve casing for all text')

    parser.add_argument('--almond_dataset_specific_preprocess', type=str, default='none', choices=['none', 'multiwoz'],
                        help='Applies dataset-sepcific preprocessing to context and answer fields, and postprocesses the model outputs back to the original form.')
    parser.add_argument("--almond_has_multiple_programs", action='store_true',
                        help='Indicate if almond dataset has multiple programs for each sentence')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of processes to use for data loading (0 means no multiprocessing)')
    
    parser.add_argument('--database_type', default='json', choices=['json', 'local-elastic', 'remote-elastic'],
                        help='database to interact with for NER')
    parser.add_argument('--elastic_config', type=str,
                        help='Path to json file containing ES configs (used for remote-elastic only)')

    parser.add_argument('--min_entity_len', type=int, default=2,
                        help='Minimum length for entities when ngrams lookup_method is used ')
    parser.add_argument('--max_entity_len', type=int, default=4,
                        help='Maximum length for entities when ngrams lookup_method is used ')
    parser.add_argument('--database_dir', type=str, help='Database folder containing all relevant files')
    
    parser.add_argument('--bootleg_input_dir', type=str, help='Path to folder containing all files (e.g. alias2qids, pretrained models) for bootleg')
    parser.add_argument('--bootleg_output_dir', type=str, default='results_temp', help='Path to folder where bootleg prepped files should be saved')
    parser.add_argument('--bootleg_model', type=str, help='Bootleg model to use')
    parser.add_argument('--bootleg_kg_encoder_layer', type=str, default=4, help='Number of kg encoder layers for BootlegBertEncoder model')
    parser.add_argument('--bootleg_dump_mode', choices=['dump_preds', 'dump_embs'], default='dump_embs',
                        help='dump_preds will dump only predictions; dump_embs will dump both prediction and embeddings')
    parser.add_argument('--bootleg_batch_size', type=int, default=30,
                        help='Batch size used for inference using bootleg')
    parser.add_argument('--bootleg_integration', type=int, choices=[1, 2],
                        help='In level 1 we extract types for top Qid candidates and feed it to the bottom of Encoder using an entity embedding layer'
                             'In level 2 we use bootleg entity embeddings directly by concatenating it with Encoder output representations')

    parser.add_argument('--lookup_method', default='ngrams', choices=['ngrams', 'smaller_first', 'longer_first'],
                        help='smaller_first: start from one token and grow into longer spans until a match is found,'
                             'longer_first: start from the longest span and shrink until a match is found')
    
    parser.add_argument('--verbose', action='store_true', help='Print detected types for each token')
    parser.add_argument('--almond_domains', nargs='+', default=[],
                        help='Domains used for almond dataset; e.g. music, books, ...')
    parser.add_argument('--features', nargs='+', type=str, default=['type', 'freq'],
                        help='Features that will be extracted for each entity: "type" and "freq" are supported.'
                             ' Order is important')
    parser.add_argument('--features_size', nargs='+', type=int, default=[1, 1],
                        help='Max length of each feature vector. All features are padded up to this length')
    parser.add_argument('--features_default_val', nargs='+', type=float, default=[0, 1.0],
                        help='Max length of each feature vector. All features are padded up to this length')
    
    parser.add_argument('--force_subword_tokenize', action='store_true', default=False,
                        help='force subword tokenization of code tokens too')
    parser.add_argument('--append_question_to_context_too', action='store_true', default=False,
                        help='')
    parser.add_argument('--override_question', default=None, help='Override the question for all tasks')
    parser.add_argument('--override_context', default=None, help='Override the context for all tasks')
    parser.add_argument('--almond_preprocess_context', action='store_true', default=False, help='')
    parser.add_argument('--almond_lang_as_question', action='store_true',
                        help='if true will use "Translate from ${language} to ThingTalk" for question')

    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    
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


def dump_bootleg_features(args, logger):
    train_sets, val_sets, aux_sets, vocab_sets = [], [], [], []
    
    train_eval_shared_kwargs = {'subsample': args.subsample, 'skip_cache': args.skip_cache,
                                'cache_input_data': args.cache_input_data,
                                'sentence_batching': args.sentence_batching,
                                'almond_lang_as_question': args.almond_lang_as_question,
                                'num_workers': args.num_workers, 'features_size': args.features_size,
                                'features_default_val': args.features_default_val,
                                'verbose': args.verbose
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
        split = task.get_splits(args.data, lower=args.lower, **kwargs)
        t1 = time.time()
        logger.info('Data loading took {} sec'.format(t1 - t0))
        assert not split.eval and not split.test
        if args.use_curriculum:
            assert split.aux
            aux_sets.append(split.aux)
            logger.info(f'{task.name} has {len(split.aux)} auxiliary examples')
        else:
            assert split.train
        train_sets.append(split.train)
        logger.info(f'{task.name} has {len(split.train)} training examples')
        
        if task.name.startswith('almond'):
            args.db_unk_id = int(args.features_default_val[0])
            if args.do_ner:
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
        split = task.get_splits(args.data, lower=args.lower, **kwargs)
        assert not split.train and not split.test and not split.aux
        logger.info(f'{task.name} has {len(split.eval)} validation examples')

    logger.info('Created bootleg features for provided datasets with subsampling: {}'.format(args.subsample))


def main(args):
    
    args.do_ner = True
    args.retrieve_method = 'bootleg'
    args.bootleg_load_prepped_data = False

    args = post_parse_general(args)
    set_seed(args)

    dump_bootleg_features(args, logger)
    
    