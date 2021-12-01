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


import asyncio
import copy
import json
import logging
import os
import sys
from pprint import pformat

import torch

from . import models
from .arguments import check_and_update_generation_args
from .calibrate import ConfidenceEstimator
from .data_utils.example import Example, NumericalizedExamples
from .ned.ned_utils import init_ned_model
from .tasks.registry import get_tasks
from .util import adjust_language_code, get_devices, load_config_json, log_model_size, set_seed
from .validate import generate_with_model

logger = logging.getLogger(__name__)

GENERATION_ARGUMENTS = {
    'num_beams',
    'num_beam_groups',
    'diversity_penalty',
    'num_outputs',
    'no_repeat_ngram_size',
    'top_p',
    'top_k',
    'repetition_penalty',
    'temperature',
    'max_output_length',
    'src_locale',
    'tgt_locale',
    'do_alignment',
    'align_preserve_input_quotation',
    'align_remove_output_quotation',
    'translate_example_split',
}


def parse_argv(parser):
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument(
        '--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used (multi-gpu currently WIP)'
    )
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--embeddings', default='.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument(
        '--checkpoint_name', default='best.pth', help='Checkpoint file to use (relative to --path, defaults to best.pth)'
    )
    parser.add_argument('--port', default=8401, type=int, help='TCP port to listen on')
    parser.add_argument('--stdin', action='store_true', help='Interact on stdin/stdout instead of TCP')
    parser.add_argument('--database_dir', type=str, help='Database folder containing all relevant files')
    parser.add_argument('--src_locale', help='locale tag of the input language to parse')
    parser.add_argument('--tgt_locale', help='locale tag of the target language to generate')
    parser.add_argument('--inference_name', default='nlp', help='name used by kfserving inference service, alphanumeric only')

    # These are generation hyperparameters. Each one can be a list of values in which case, we generate `num_outputs` outputs for each set of hyperparameters.
    parser.add_argument("--num_outputs", type=int, nargs='+', default=[1], help='number of sequences to output per input')
    parser.add_argument("--temperature", type=float, nargs='+', default=[0.0], help="temperature of 0 implies greedy sampling")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        nargs='+',
        default=[1.0],
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[1.0], help='1.0 disables top-p filtering')
    parser.add_argument("--num_beams", type=int, nargs='+', default=[1], help='1 disables beam seach')
    parser.add_argument("--num_beam_groups", type=int, nargs='+', default=[1], help='1 disables diverse beam seach')
    parser.add_argument("--diversity_penalty", type=float, nargs='+', default=[0.0], help='0 disables diverse beam seach')
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        nargs='+',
        default=[0],
        help='ngrams of this size cannot be repeated in the output. 0 disables it.',
    )
    parser.add_argument('--max_output_length', default=150, type=int, help='maximum output length for generation')

    # for confidence estimation:
    parser.add_argument(
        '--calibrator_paths',
        type=str,
        nargs='+',
        default=None,
        help='If provided, will be used to output confidence scores for each prediction. Defaults to `--path`/calibrator.pkl',
    )


class Server(object):
    def __init__(self, args, numericalizer, model, device, confidence_estimators, estimator_filenames, ned_model):
        self.args = args
        self.device = device
        self.numericalizer = numericalizer
        self.model = model
        self.confidence_estimators = confidence_estimators
        self.estimator_filenames = estimator_filenames
        self.ned_model = ned_model

        self._cached_task_names = dict()

    def numericalize_examples(self, ex):

        all_features = NumericalizedExamples.from_examples(ex, self.numericalizer)
        # make a single batch with all examples
        return NumericalizedExamples.collate_batches(all_features, self.numericalizer, device=self.device)

    def _init_request(self, request):
        args = copy.deepcopy(self.args)
        generation_options = request.get('options', {})
        for k, v in generation_options.items():
            if k not in GENERATION_ARGUMENTS:
                logger.warning(f'{k} is not a generation option and cannot be overriden')
                continue
            setattr(args, k, v)

        # TODO handle this better by decoupling numericalizer and model
        if hasattr(args, 'src_locale') and hasattr(args, 'tgt_locale'):
            src_locale, tgt_locale = adjust_language_code(
                self.model.config, self.args.pretrained_model, args.src_locale, args.tgt_locale
            )
            self.numericalizer.update_language_dependent_properties(src_locale, tgt_locale)
            self.model.update_language_dependent_configs(tgt_locale)

        task_name = request['task'] if 'task' in request else 'generic'
        task = list(get_tasks([task_name], args, self._cached_task_names).values())[0]
        if task_name not in self._cached_task_names:
            self._cached_task_names[task_name] = task

        return task, args

    def _numericalize_request(self, request, task, args):
        # if single example wrap it as a list
        if 'instances' not in request:
            request['instances'] = [
                {
                    'example_id': request.get('example_id', ''),
                    'context': request['context'],
                    'question': request['question'],
                    'answer': request.get('answer', ''),
                }
            ]

        examples = []
        # request['instances'] is an array of {context, question, answer, example_id}
        for instance in request['instances']:
            example_id, context, question, answer = (
                instance.get('example_id', ''),
                instance['context'],
                instance['question'],
                instance.get('answer', ''),
            )
            if not context:
                context = task.default_context
            if not question:
                question = task.default_question

            ex = Example.from_raw(
                str(example_id), context, question, answer, preprocess=task.preprocess_field, lower=args.lower
            )
            examples.append(ex)

        # process features for examples
        if self.ned_model:
            self.ned_model.process_examples(examples, None, task.utterance_field)

        self.model.add_new_vocab_from_data([task])
        self.model.set_generation_output_options([task])

        return self.numericalize_examples(examples)

    def _predict_batch(self, batch, task, args):
        if args.calibrator_paths is not None:
            output = generate_with_model(
                self.model,
                [batch],
                self.numericalizer,
                task,
                args,
                output_predictions_only=True,
                confidence_estimators=self.confidence_estimators,
            )
            response = []
            if sum(args.num_outputs) > 1:
                for idx, predictions in enumerate(output.predictions):
                    candidates = []
                    for cand in predictions:
                        candidate = {'answer': cand, 'score': {}}
                        for e_idx, estimator_scores in enumerate(output.confidence_scores):
                            candidate['score'][self.estimator_filenames[e_idx]] = float(estimator_scores[idx])
                        candidates.append(candidate)
                    response.append({'candidates': candidates})
            else:
                for idx, p in enumerate(output.predictions):
                    instance = {'answer': p[0], 'score': {}}
                    for e_idx, estimator_scores in enumerate(output.confidence_scores):
                        instance['score'][self.estimator_filenames[e_idx]] = float(estimator_scores[idx])
                    response.append(instance)
        else:
            output = generate_with_model(
                self.model, [batch], self.numericalizer, task, args, output_predictions_only=True
            )
            if sum(args.num_outputs) > 1:
                response = []
                for idx, predictions in enumerate(output.predictions):
                    candidates = []
                    for cand in predictions:
                        candidates.append({'answer': cand})
                    response.append({'candidates': candidates})
            else:
                response = [{'answer': p[0]} for p in output.predictions]

        return response

    def handle_request(self, request):
        try:
            with torch.no_grad():
                task, args = self._init_request(request)
                batch = self._numericalize_request(request, task, args)
                response = self._predict_batch(batch, task, args)
        except RuntimeError as e:
            # catch all cuda errors and exit
            if 'CUDA error' in str(e):
                exit(100)
            else:
                raise e

        return response

    def handle_json_request(self, line: str) -> str:
        request = json.loads(line)
        if 'instances' in request:
            return json.dumps({'id': request['id'], 'instances': self.handle_request(request)}) + '\n'
        else:
            response = self.handle_request(request)
            assert len(response) == 1
            response = response[0]
            response['id'] = request['id']
            return json.dumps(response, ensure_ascii=False) + '\n'

    async def handle_client(self, client_reader, client_writer):
        try:
            line = await client_reader.readline()
            while line:
                client_writer.write(self.handle_json_request(line).encode('utf-8'))
                line = await client_reader.readline()

        except IOError:
            logger.info('Connection to client_reader closed')
            try:
                client_writer.close()
            except IOError:
                pass

    def _run_tcp(self):
        loop = asyncio.get_event_loop()
        server = loop.run_until_complete(asyncio.start_server(self.handle_client, port=self.args.port))
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.close()

    def _run_stdin(self):
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                sys.stdout.write(self.handle_json_request(line))
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass

    def run(self):
        log_model_size(logger, self.model, self.args.model)
        self.model.to(self.device)

        self.model.eval()
        if self.args.stdin:
            self._run_stdin()
        else:
            self._run_tcp()


def init(args):
    load_config_json(args)
    check_and_update_generation_args(args)
    if not args.src_locale:
        args.src_locale = args.eval_src_languages
    if not args.tgt_locale:
        args.tgt_locale = args.eval_tgt_languages
    set_seed(args)

    devices = get_devices()
    device = devices[0]  # server only runs on a single device

    if args.ned_retrieve_method == 'bootleg':
        ned_model = init_ned_model(args, 'bootleg-annotator')
    else:
        ned_model = init_ned_model(args)

    logger.info(f'Arguments:\n{pformat(vars(args))}')
    logger.info(f'Loading from {args.best_checkpoint}')

    Model = getattr(models, args.model)
    model, _ = Model.load(
        args.path,
        model_checkpoint_file=args.checkpoint_name,
        args=args,
        device=device,
        src_lang=args.src_locale,
        tgt_lang=args.tgt_locale,
    )

    model.to(device)
    model.eval()

    # set the default path for calibrator if it exists
    estimator_filenames = []
    if args.calibrator_paths is None:
        for filename in os.listdir(args.path):
            path = os.path.join(args.path, filename)
            if not ConfidenceEstimator.is_estimator(path):
                continue
            if args.calibrator_paths is None:
                args.calibrator_paths = []
            args.calibrator_paths.append(path)
            estimator_filenames.append(os.path.splitext(filename)[0])

    confidence_estimators = None
    if args.calibrator_paths is not None:
        confidence_estimators = []
        for path in args.calibrator_paths:
            estimator = ConfidenceEstimator.load(path)
            confidence_estimators.append(estimator)
            logger.info('Loading confidence estimator "%s" from %s', estimator.name, path)
        args.mc_dropout_num = confidence_estimators[0].mc_dropout_num  # we assume all estimators have the same mc_dropout_num

    return model, device, confidence_estimators, estimator_filenames, ned_model


def main(args):
    model, device, confidence_estimators, estimator_filenames, ned_model = init(args)
    server = Server(args, model.numericalizer, model, device, confidence_estimators, estimator_filenames, ned_model)
    server.run()
