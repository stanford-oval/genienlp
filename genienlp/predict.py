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

from .data_utils.bootleg import Bootleg, init_bootleg_annotator, extract_features_with_annotator
from .models import TransformerForTokenClassification
from .run_bootleg import bootleg_process_splits

try:
     set_start_method('spawn')
except RuntimeError:
    pass

import torch

from . import models
from .tasks.registry import get_tasks
from .util import set_seed, load_config_json, make_data_loader, log_model_size, get_devices, \
    combine_folders_on_disk, split_folder_on_disk, get_part_path
from .validate import calculate_and_reduce_metrics, generate_with_model
from .calibrate import ConfidenceEstimator
from .arguments import check_and_update_generation_args

logger = logging.getLogger(__name__)



def prepare_data(args, device):
    # initialize bootleg
    bootleg = None
    if args.do_ned and args.ned_retrieve_method == 'bootleg':
        bootleg = Bootleg(args)
    
    datasets = []
    paths = []
    if len(args.pred_src_languages) == 1 and len(args.tasks) > 1:
        args.pred_src_languages *= len(args.tasks)
    for i, task in enumerate(args.tasks):
        task_languages = args.pred_src_languages[i]
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
        
        kwargs.update({'skip_cache': args.skip_cache,
                       'subsample': args.subsample,
                       'cached_path': os.path.join(args.cache, task.name),
                       'all_dirs': task_languages,
                       'almond_lang_as_question': args.almond_lang_as_question,
                       'num_workers': args.num_workers,
                       'separate_eval': args.separate_eval,
                       'translate_no_answer': args.translate_no_answer,
                       'ner_domains': args.ner_domains,
                       'hf_test_overfit': args.hf_test_overfit
                       })
        
        task_splits, task_paths = task.get_splits(root=args.data, lower=args.lower, **kwargs)
        if not isinstance(task_splits, list):
            task_splits = [task_splits]
            task_paths = [task_paths]
        task_data_processed = []
        task_path_processed = []
        for split, path in zip(task_splits, task_paths):
            assert (split.eval or split.test or split.train) and not split.aux
            if split.train:
                data = split.train
                path = path.train
            elif split.eval:
                data = split.eval
                path = path.eval
            else:
                data = split.test
                path = path.test
            if bootleg:
                if split.train or split.eval:
                    bootleg_process_splits(args, data.examples, path, task, bootleg)
                else:
                    # no prepped bootleg features are available
                    # extract features on-the-fly using bootleg annotator
                    bootleg_annotator = init_bootleg_annotator(args, device)
                    extract_features_with_annotator(data.examples, bootleg_annotator, args, task)
            task_data_processed.append(data)
            task_path_processed.append(path)
        datasets.append(task_data_processed)
        paths.append(task_path_processed)

    return datasets


def prepare_data_iterators(args, val_sets, numericalizer, device):
    logger.info(f'Preparing data iterators')
    if len(args.val_batch_size) == 1 and len(val_sets) > 1:
        args.val_batch_size *= len(val_sets)
    iters = []
    task_index = 0
    for task, bs, val_set in zip(args.tasks, args.val_batch_size, val_sets):
        task_iter = []
        task_languages = args.pred_src_languages[task_index]
        if task_languages is not None and args.separate_eval:
            task_languages = task_languages.split('+')
            assert len(task_languages) == len(val_set)
            for index, set_ in enumerate(val_set):
                loader, original_order = make_data_loader(set_, numericalizer, bs, device,
                                                          train=False, return_original_order=True)
                task_iter.append((task, task_languages[index], loader, original_order))
        # single language task or no separate eval
        else:
           loader, original_order = make_data_loader(val_set[0], numericalizer, bs, device,
                                                     train=False, return_original_order=True)
           task_iter.append((task, task_languages, loader, original_order))

        iters.extend(task_iter)
        task_index += 1

    return iters


def run(args, device):
    
    # TODO handle multiple languages
    src_lang = args.pred_src_languages[0]
    tgt_lang = args.pred_tgt_languages[0]

    Model = getattr(models, args.model)
    model, _ = Model.load(args.path,
                                     model_checkpoint_file=args.checkpoint_name,
                                     args=args,
                                     device=device,
                                     tasks=args.tasks,
                                     src_lang=src_lang,
                                     tgt_lang=tgt_lang
                                     )

    val_sets = prepare_data(args, device)
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
            if language is None or 'multilingual' not in task.name:
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

            if args.calibrator_paths is not None:
                confidence_estimators = []
                for path in args.calibrator_paths:
                    estimator = ConfidenceEstimator.load(path)
                    confidence_estimators.append(estimator)
                    logger.info('Loading confidence estimator "%s" from %s', estimator.name, path)
            else:
                confidence_estimators = None
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                generation_output = generate_with_model(model, it, model.numericalizer, task, args,
                                                                     original_order=original_order,
                                                                     output_confidence_features=args.save_confidence_features,
                                                                     confidence_estimators=confidence_estimators,
                                                                     disable_progbar=False)

            if args.save_confidence_features:
                torch.save(generation_output.confidence_features, args.confidence_feature_path)

            # write into file
            # TODO change to jsonl format
            with open(prediction_file_name, 'w' + ('' if args.overwrite else 'x')) as prediction_file:
                for i in range(len(generation_output.example_ids)):
                    line = generation_output.example_ids[i] + '\t' + '\t'.join(generation_output.predictions[i]) # all outputs separated by '\t'
                    if args.calibrator_paths is not None:
                        for score in generation_output.confidence_scores:
                            line += '\t' + str(score[i])
                    prediction_file.write(line + '\n')

            if len(generation_output.answers) > 0:
                metrics_to_compute = task.metrics
                if args.main_metric_only:
                    metrics_to_compute = [metrics_to_compute[0]]
                metrics = calculate_and_reduce_metrics(generation_output.predictions, generation_output.answers, metrics_to_compute, args.reduce_metrics)

                with open(results_file_name, 'w' + ('' if args.overwrite else '+')) as results_file:
                    results_file.write(json.dumps(metrics) + '\n')

                if not args.silent:
                    for i, (c, p, a) in enumerate(zip(generation_output.contexts, generation_output.predictions, generation_output.answers)):
                        log_string = f'\nContext {i+1}: {c}\nPrediction {i + 1} ({len(p)} outputs): {p}\nAnswer {i + 1}: {a}\n'
                        if args.calibrator_paths is not None:
                            log_string += f'Confidence {i+1} : '
                            for score in generation_output.confidence_scores:
                                log_string += f'{score[i]:.3f}, '
                            log_string += '\n'
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
    parser.add_argument('--tasks', dest='task_names', nargs='+', help='task names for prediction')
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
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the eval/test datasets')
                        
    parser.add_argument('--pred_languages', type=str, nargs='+', dest='pred_src_languages', default=['en'],
                        help='Specify dataset source languages used during prediction for multilingual tasks'
                        'multiple languages for each task should be concatenated with +')
    parser.add_argument('--pred_tgt_languages', type=str, nargs='+', default=['en'],
                        help='Specify dataset target languages used during prediction for multilingual tasks'
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
    parser.add_argument('--calibrator_paths', type=str, nargs='+', default=None,
                        help='Can be a list. If provided, each calibrator will be used to output confidence scores for each prediction.')
    parser.add_argument('--save_confidence_features', action='store_true', help='If provided, will be used to output confidence scores for each prediction.')
    parser.add_argument("--confidence_feature_path", type=str, default=None, help='A .pkl file to save confidence features in.')
    parser.add_argument("--mc_dropout_num", type=int, default=0, help='Number of samples to use for Monte Carlo (MC) dropout. 0 disables MC dropout.')
    parser.add_argument("--override_confidence_labels", type=str, default=None,
                        help='If provided, examples with this gold answer are marked as 1, and others as 0. Useful for out-of-domain detection.')

    parser.add_argument('--database_dir', type=str, help='Path to folder containing all files (e.g. alias2qids, pretrained models for bootleg)')

    parser.add_argument("--mixed_precision", action='store_true', help='If True, will use mixed precision for prediction.'
                        'This reduces memory consumption and is especially faster on GPUs like NVIDIA V100 and T4. May slightly change the generated output.')
    
    #TODO Update other tasks to use this argument too; so we can use predict for pure text generation (i.e. without reporting accuracy metrics)
    parser.add_argument('--translate_no_answer', action='store_true', help='if true the provided dataset would not contain the answer (translated sentence)')
    parser.add_argument('--plot_heatmaps', action='store_true', help='whether to plot cross-attention heatmaps')

            
def set_default_values(args):
    """
    sets default values that depend on other input arguments
    """
    if args.confidence_feature_path is None:
        args.confidence_feature_path = os.path.join(args.path, 'confidence_features.pkl')


def check_args(args):
    
    if len(args.task_names) != len(args.pred_src_languages):
        raise ValueError('You have to define prediction languages for each task'
                         'Use None for single language tasks. Also provide languages in the same order you provided the tasks.')

    if getattr(args, 'do_ned', False) and getattr(args, 'ned_retrieve_method', None) == 'bootleg':
        with open(os.path.join(args.path, 'config.json')) as config_file:
            config = json.load(config_file)
        if args.subsample > config['subsample']:
            raise ValueError('To use bootleg, you have to use a subsample value less than the number of prepped examples.')
            

def main(args):
    load_config_json(args)
    check_and_update_generation_args(args)
    check_args(args)
    set_default_values(args)

    set_seed(args)
    args.tasks = list(get_tasks(args.task_names, args).values())

    logger.info(f'Arguments:\n{pformat(vars(args))}')
    logger.info(f'Loading from {args.best_checkpoint}')

    devices = get_devices(args.devices)

    if len(devices) > 1:
        logger.info(f'Independent multi-GPU generation on following devices: {devices}')
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
        logger.info(f'Single device generation on: {devices[0]}')
        run(args, devices[0])
        
