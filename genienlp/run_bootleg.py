import os
import datetime
from pprint import pformat
import logging

from .arguments import save_args, get_commit
from .tasks.registry import get_tasks

from .train import initialize_logger
from .util import set_seed, have_multilingual
import time

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
    
    parser.add_argument('--bootleg_input_dir', type=str,
                        help='Path to folder containing all files (e.g. alias2qids, pretrained models) for bootleg')
    parser.add_argument('--bootleg_model', type=str, help='Bootleg model to use')
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


def post_parse(args):
    
    args.bootleg_dump_features = True
    args.retrieve_method = 'bootleg'
    args.do_ner = True
    
    if args.val_task_names is None:
        args.val_task_names = []
        for t in args.train_task_names:
            if t not in args.val_task_names:
                args.val_task_names.append(t)
    if 'imdb' in args.val_task_names:
        args.val_task_names.remove('imdb')
    
    args.timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    
    def indices_of_multilingual(train_task_names):
        indices = []
        for i, task in enumerate(train_task_names):
            if 'multilingual' in task:
                indices.append(i)
        return indices
    
    if args.sentence_batching and args.train_batch_size == 0:
        raise ValueError('You need to specify train_batch_size value when using sentence batching.')
    # TODO relax the following assertions by dropping samples from batches in Iter
    if args.sentence_batching and args.train_batch_size % len(args.train_languages.split('+')) != 0:
        raise ValueError(
            'Your train_batch_size should be divisible by number of train_languages when using sentence batching.')
    if args.sentence_batching and args.val_batch_size[0] % len(args.eval_languages.split('+')) != 0:
        raise ValueError(
            'Your val_batch_size should be divisible by number of eval_languages when using sentence batching.')
    
    if len(args.features) != len(args.features_size):
        raise ValueError('You should specify max feature size for each feature you provided')
    
    if args.override_context and args.append_question_to_context_too:
        raise ValueError('You cannot use append_question_to_context_too when overriding context')
    
    if args.paired and not args.sentence_batching:
        logger.warning('Paired training only works if sentence_batching is used as well.'
                       'Activating sentence_batching...')
        args.sentence_batching = True
    
    if args.min_entity_len <= 0:
        logger.warning('min_entity_len should be equal to or greater than 1')
    
    args.train_batch_values = args.train_batch_tokens
    if len(args.train_task_names) > 1:
        if args.train_iterations is None:
            args.train_iterations = [1]
        if len(args.train_iterations) < len(args.train_task_names):
            args.train_iterations = len(args.train_task_names) * args.train_iterations
        if len(args.train_batch_tokens) < len(args.train_task_names):
            args.train_batch_values = len(args.train_task_names) * args.train_batch_tokens
    indices = indices_of_multilingual(args.train_task_names)
    for i in indices:
        if args.sentence_batching:
            args.train_batch_values[i] = args.train_batch_size
            if args.paired:
                num_train_langs = len(args.train_languages.split('+'))
                new_batch_size = int(args.train_batch_size * \
                                     (1 + min(num_train_langs ** 2 - num_train_langs,
                                              args.max_pairs) / num_train_langs))
                logger.warning('Using paired example training will increase effective batch size from {} to {}'.
                               format(args.train_batch_size, new_batch_size))
    
    if len(args.val_batch_size) < len(args.val_task_names):
        args.val_batch_size = len(args.val_task_names) * args.val_batch_size
    
    # postprocess arguments
    if args.commit:
        args.commit = get_commit()
    else:
        args.commit = ''
    
    if have_multilingual(args.train_task_names) and (args.train_languages is None or args.eval_languages is None):
        raise ValueError('You have to define training and evaluation languages when you have a multilingual task')
    
    args.log_dir = args.save
    args.dist_sync_file = os.path.join(args.log_dir, 'distributed_sync_file')
    
    for x in ['data', 'log_dir', 'dist_sync_file']:
        setattr(args, x, os.path.join(args.root, getattr(args, x)))
    
    args.num_features = len(args.features)
    
    # tasks with the same name share the same task object
    train_tasks_dict = get_tasks(args.train_task_names, args)
    args.train_tasks = list(train_tasks_dict.values())
    val_task_dict = get_tasks(args.val_task_names, args, available_tasks=train_tasks_dict)
    args.val_tasks = list(val_task_dict.values())
    
    save_args(args)
    
    return args


def main(args):
    args = post_parse(args)
    set_seed(args)

    dump_bootleg_features(args, logger)
    
    