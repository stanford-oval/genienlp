#
# Copyright (c) 2018, Salesforce, Inc.
#                     The Board of Trustees of the Leland Stanford Junior University
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

import json
import logging
import os
from pprint import pformat
from collections import defaultdict
import copy
import shutil

# multiprocessing with CUDA
from torch.multiprocessing import Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import torch
import pickle

from . import models
from .tasks.registry import get_tasks
from .util import set_seed, load_config_json, make_data_loader, log_model_size, init_devices, \
    have_multilingual, combine_folders_on_disk, split_folder_on_disk, get_part_path
from .validate import generate_with_model, calculate_and_reduce_metrics
from .calibrate import ConfidenceEstimator
from .arguments import check_and_update_generation_args

logger = logging.getLogger(__name__)

def get_all_splits(args):
    splits = []
    if len(args.pred_languages) == 1 and len(args.tasks) > 1:
        args.pred_languages *= len(args.tasks)
    for i, task in enumerate(args.tasks):
        task_languages = args.pred_languages[i]
        logger.info(f'Loading {task}')
        kwargs = {'train': None, 'validation': None, 'test': None}
        if args.evaluate == 'train':
            del kwargs['train'] # deleting keys means use the default file name
        elif args.evaluate == 'valid':
            kwargs['validation'] = args.pred_set_name
        elif args.evaluate == 'test':
            del kwargs['test']
        else:
            raise ValueError('Split used for prediction should be either train, valid or test')
        
        kwargs.update({'skip_cache': args.skip_cache, 'subsample': args.subsample,
                       'cached_path': os.path.join(args.cache, task.name), 'all_dirs': task_languages,
                       'almond_lang_as_question': args.almond_lang_as_question})
        
        kwargs['separate_eval'] = args.separate_eval
        task_splits = task.get_splits(root=args.data, lower=args.lower, **kwargs)
        if not isinstance(task_splits, list):
            task_splits = [task_splits]
        task_split_processed = []
        for split in task_splits:
            assert (split.eval or split.test or split.train) and not split.aux
            if split.train:
                split = split.train
            elif split.eval:
                split = split.eval
            else:
                split = split.test
            task_split_processed.append(split)
        splits.append(task_split_processed)
    return splits


def prepare_data_iterators(args, val_sets, numericalizer, device):
    logger.info(f'Preparing data iterators')
    if len(args.val_batch_size) == 1 and len(val_sets) > 1:
        args.val_batch_size *= len(val_sets)
    iters = []
    task_index = 0
    for task, bs, val_set in zip(args.tasks, args.val_batch_size, val_sets):
        task_iter = []
        task_languages = args.pred_languages[task_index]
        if task_languages is not None and args.separate_eval:
            task_languages = task_languages.split('+')
            assert len(task_languages) == len(val_set)
            for index, set_ in enumerate(val_set):
                loader, original_order = make_data_loader(set_, numericalizer, bs, device, train=False, return_original_order=True)
                task_iter.append((task, task_languages[index], loader, original_order))
        # single language task or no separate eval
        else:
           loader, original_order = make_data_loader(val_set[0], numericalizer, bs, device, train=False, return_original_order=True)
           task_iter.append((task, task_languages, loader, original_order))

        iters.extend(task_iter)
        task_index += 1

    return iters


def run(args, device):
    Model = getattr(models, args.model)
    model, _ = Model.from_pretrained(args.path,
                                     model_checkpoint_file=args.checkpoint_name,
                                     args=args,
                                     device=device,
                                     tasks=args.tasks,
                                    )
    
    if args.pred_languages[0] is not None:
        model.set_decoder_start_token_id(args.pred_languages[0].split('+')[0])
    else:
        # use English as default
        model.set_decoder_start_token_id('en')

    val_sets = get_all_splits(args)
    model.add_new_vocab_from_data(args.tasks)

    iters = prepare_data_iterators(args, val_sets, model.numericalizer, device)

    log_model_size(logger, model, args.model)
    model.to(device)

    decaScore = []
    task_scores = defaultdict(list)
    model.eval()

    eval_dir = os.path.join(args.eval_dir, args.evaluate)
    os.makedirs(eval_dir, exist_ok=True)

    with torch.no_grad():
        for task, language, it, original_order in iters:
            logger.info(task.name)
            # single language task
            if language is None:
                prediction_file_name = os.path.join(eval_dir, task.name + '.tsv')
                results_file_name = os.path.join(eval_dir, task.name + '.results.json')
            # multi language task
            else:
                prediction_file_name = os.path.join(eval_dir, task.name + '_{}.tsv'.format(language))
                results_file_name = os.path.join(eval_dir, task.name + '_{}.results.json'.format(language))
            if os.path.exists(prediction_file_name):
                if args.overwrite:
                    logger.warning(f'{prediction_file_name} already exists -- overwriting **')
                else:
                    raise OSError(f'{prediction_file_name} already exists')
            if os.path.exists(results_file_name):
                if args.overwrite:
                    logger.warning(f'{results_file_name} already exists -- overwriting **')
                else:
                    raise OSError(f'{results_file_name} already exists')

            if args.calibrator_path is not None:
                confidence_estimator = ConfidenceEstimator.load(args.calibrator_path)
                logger.info('Loading confidence estimator "%s" from %s', confidence_estimator.name, args.calibrator_path)
            else:
                confidence_estimator = None
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                generation_output = generate_with_model(model, it, model.numericalizer, task, args,
                                                     original_order=original_order,
                                                     output_confidence_features=args.save_confidence_features,
                                                     confidence_estimator=confidence_estimator)
            
            if args.save_confidence_features:
                with open(args.confidence_feature_path, 'wb') as f:
                    pickle.dump(generation_output.confidence_features, f, protocol=4)

            # write into file
            # TODO change to jsonl format
            with open(prediction_file_name, 'w' + ('' if args.overwrite else 'x')) as prediction_file:
                for i in range(len(generation_output.example_ids)):
                    line = generation_output.example_ids[i] + '\t' + '\t'.join(generation_output.predictions[i]) # all outputs separated by '\t'
                    if args.calibrator_path is not None:
                        line += '\t' + str(generation_output.confidence_scores[i])
                    prediction_file.write(line + '\n')

            if len(generation_output.answers) > 0:
                metrics_to_compute = task.metrics
                if args.main_metric_only:
                    metrics_to_compute = [metrics_to_compute[0]]
                metrics = calculate_and_reduce_metrics(generation_output.predictions, generation_output.answers, metrics_to_compute, args)

                with open(results_file_name, 'w' + ('' if args.overwrite else '+')) as results_file:
                    results_file.write(json.dumps(metrics) + '\n')

                if not args.silent:
                    for i, (c, p, a) in enumerate(zip(generation_output.contexts, generation_output.predictions, generation_output.answers)):
                        log_string = f'\nContext {i+1}: {c}\nPrediction {i + 1} ({len(p)} outputs): {p}\nAnswer {i + 1}: {a}\n'
                        if args.calibrator_path is not None:
                            log_string += f'Confidence {i+1} : {generation_output.confidence_scores[i]:.3f}\n'
                        logger.info(log_string)
                    logger.info(metrics)
                    
                task_scores[task].append((len(generation_output.answers), metrics[task.metrics[0]]))
    
    for task in task_scores.keys():
        decaScore.append(sum([length * score for length, score in task_scores[task]]) / sum([length for length, score in task_scores[task]]))

    logger.info(f'Evaluated Tasks:\n')
    for i, task in enumerate(args.tasks):
        logger.info(f'{task.name}: {decaScore[i]}')
    logger.info(f'-------------------')
    logger.info(f'DecaScore:  {sum(decaScore)}\n')
    logger.info(f'\nSummary: | {sum(decaScore)} | {" | ".join([str(x) for x in decaScore])} |\n')


def parse_argv(parser):
    parser.add_argument('--path', type=str, required=True, help='Folder to load the model from')
    parser.add_argument('--evaluate', type=str, required=True, choices=['train', 'valid', 'test'],
                        help='Which dataset to do predictions for (train, dev or test)')
    parser.add_argument('--pred_set_name', default='eval', type=str, help='Name of dataset to run prediction for; will be ignored if --evaluate is test')
    parser.add_argument('--tasks',
                        default=['almond', 'squad', 'iwslt.en.de', 'cnn_dailymail', 'multinli.in.out', 'sst', 'srl',
                                 'zre', 'woz.en', 'wikisql', 'schema'], dest='task_names', nargs='+')
    parser.add_argument('--devices', default=None, nargs='+', type=int,
                        help='a list of devices that can be used for prediction. By default, all devices will be used.')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--data', default='.data/', type=str, help='where to load data from.')
    parser.add_argument('--embeddings', default='.embeddings/', type=str, help='where to save embeddings.')
    parser.add_argument('--checkpoint_name', default='best.pth',
                        help='Checkpoint file to use (relative to --path, defaults to best.pth)')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite previously written predictions')
    parser.add_argument('--silent', action='store_true', help='whether to print predictions to stdout')

    parser.add_argument('--skip_cache', action='store_true',
                        help='whether use exisiting cached splits or generate new ones')
    parser.add_argument('--eval_dir', type=str, required=True, help='use this directory to store eval results')
    parser.add_argument('--cache', default='.cache', type=str, help='where to save cached files')
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the eval/test datasets (experimental)')
                        
    parser.add_argument('--pred_languages', type=str, nargs='+',
                        help='used to specify dataset languages used during prediction for multilingual tasks'
                        'multiple languages for each task should be concatenated with +')
    parser.add_argument('--separate_eval', action='store_true',
                        help='evaluate on each language eval set separately')
    
    parser.add_argument('--main_metric_only', action='store_true', help='If True, we only calculate the deca score metric for each task.')
    # If not None, these values will override the values saved in the trained model's config file
    parser.add_argument('--val_batch_size', nargs='+', default=None, type=int,
                        help='Batch size for validation corresponding to tasks in val tasks')
    parser.add_argument("--reduce_metrics", type=str, default='max', choices=['max'], help='How to calculate the metric when there are multiple outputs per input.')

    # These are generation hyperparameters. Each one can be a list of values in which case, we generate `num_outputs` outputs for each set of hyperparameters.
    parser.add_argument("--num_outputs", type=int, nargs='+', default=[1], help='number of sequences to output per input')
    parser.add_argument("--temperature", type=float, nargs='+', default=[0.0],
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, nargs='+', default=[1.0],
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[1.0], help='1.0 disables top-p filtering')
    parser.add_argument("--num_beams", type=int, nargs='+', default=[1], help='1 disables beam seach')
    parser.add_argument("--num_beam_groups", type=int, nargs='+', default=[1], help='1 disables diverse beam seach')
    parser.add_argument("--diversity_penalty", type=float, nargs='+', default=[0.0], help='0 disables diverse beam seach')
    parser.add_argument("--no_repeat_ngram_size", type=int, nargs='+', default=[0], help='ngrams of this size cannot be repeated in the output. 0 disables it.')
    parser.add_argument('--max_output_length', default=150, type=int, help='maximum output length for generation')

    # These are used for confidence calibration
    parser.add_argument('--calibrator_path', type=str, default=None, help='If provided, will be used to output confidence scores for each prediction.')
    parser.add_argument('--save_confidence_features', action='store_true', help='If provided, will be used to output confidence scores for each prediction.')
    parser.add_argument("--confidence_feature_path", type=str, default=None, help='A .pkl file to save confidence features in.')
    parser.add_argument("--mc_dropout", action='store_true', help='Monte Carlo dropout')
    parser.add_argument("--mc_dropout_num", type=int, default=0, help='Number of samples to use for Monte Carlo dropout')

    parser.add_argument("--mixed_precision", action='store_true', help='If True, will use mixed precision for prediction.'
                        'This reduces memory consumption and is especially faster on GPUs like NVIDIA V100 and T4. May slightly change the generated output.')


def adjust_multilingual_eval(args):
    if (have_multilingual(args.task_names) and args.pred_languages is None) or (
            args.pred_languages and len(args.task_names) != len(args.pred_languages)):
        raise ValueError('You have to define prediction languages when you have a multilingual task'
                         'Use None for single language tasks. Also provide languages in the same order you provided the tasks.')

    if args.pred_languages is None:
        args.pred_languages = [None for _ in range(len(args.task_names))]
        
    if 'mbart' in args.pretrained_model:
        if args.pred_languages[0] and len(args.pred_languages[0].split('+')) != 1:
            raise ValueError('For now we only support single language prediction with mbart models')

    # preserve backward compatibility for single language tasks
    for i, task_name in enumerate(args.task_names):
        if 'multilingual' in task_name and args.pred_languages[i] is None:
            raise ValueError('You have to define prediction languages for this multilingual task: {}'.format(task_name))
        elif 'multilingual' not in task_name and args.pred_languages[i] is not None:
            logger.warning('prediction languages should be empty for single language tasks')
            args.pred_languages[i] = None
            
            
def set_default_values(args):
    """
    sets default values that depend on other input arguments
    """
    if args.confidence_feature_path is None:
        args.confidence_feature_path = os.path.join(args.path, 'confidence_features.pkl')


def main(args):
    load_config_json(args)
    check_and_update_generation_args(args)
    adjust_multilingual_eval(args)
    set_default_values(args)

    set_seed(args)
    args.tasks = list(get_tasks(args.task_names, args).values())

    logger.info(f'Arguments:\n{pformat(vars(args))}')
    logger.info(f'Loading from {args.best_checkpoint}')

    devices = init_devices(args)
    if args.devices is not None:
        devices = [devices[i] for i in args.devices]

    if len(devices) > 1:
        # Independent multi-GPU generation
        all_processes = []
        all_data_folders = split_folder_on_disk(args.data, len(devices))
        
        for device_id in range(len(devices)):
            copy_args = copy.copy(args)
            copy_args.data = all_data_folders[device_id]
            copy_args.eval_dir = get_part_path(args.eval_dir, device_id)
            
            p = Process(target=run, args=(copy_args, devices[device_id]))
            all_processes.append(p)
            p.start()

        for p in all_processes:
            p.join()

        for folder in all_data_folders:
            shutil.rmtree(folder)
        combine_folders_on_disk(args.eval_dir, len(devices), line_group_size=1, delete=True)

    else:
        run(args, devices[0])
        
