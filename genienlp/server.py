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
from genienlp.calibrate import ConfidenceEstimator
import json
import logging
import sys
import os
from pprint import pformat

import torch

from . import models
from .data_utils.example import Example, NumericalizedExamples
from .tasks.registry import get_tasks
from .util import set_seed, init_devices, load_config_json, log_model_size
from .validate import generate_with_model

from bootleg.annotator import Annotator
from .data_utils.bootleg import Bootleg

logger = logging.getLogger(__name__)


class Server:
    def __init__(self, args, numericalizer, model, device, confidence_estimator, bootleg_annotator=None):
        self.args = args
        self.device = device
        self.numericalizer = numericalizer
        self.model = model
        self.confidence_estimator = confidence_estimator
        self.bootleg_annotator = bootleg_annotator

        self._cached_task_names = dict()

    def numericalize_examples(self, ex):

        all_features = NumericalizedExamples.from_examples(ex, self.numericalizer, self.args.add_types_to_text)
        # make a single batch with all examples
        return NumericalizedExamples.collate_batches(all_features, self.numericalizer, device=self.device, db_unk_id=self.args.db_unk_id)
    
    def bootleg_process_examples(self, ex, label, task):
        line = {}
        if task.is_contextual():
            line['sentence'] = ex.question
        else:
            line['sentence'] = ex.context
    
        assert len(label) == 7
        line['cands'] = label[3]
        line['cand_probs'] = list(map(lambda item: list(item), label[4]))
        line['spans'] = label[5]
        line['aliases'] = label[6]
        tokens_type_ids, tokens_type_probs = self.bootleg_annotator.bootleg.collect_features_per_line(line, self.args.bootleg_prob_threshold)
    
        if task.is_contextual():
            for i in range(len(tokens_type_ids)):
                ex.question_feature[i].type_id = tokens_type_ids[i]
                ex.question_feature[i].type_prob = tokens_type_probs[i]
                ex.context_plus_question_feature[i + len(ex.context.split(' '))].type_id = tokens_type_ids[i]
                ex.context_plus_question_feature[i + len(ex.context.split(' '))].type_prob = tokens_type_probs[i]
    
        else:
            for i in range(len(tokens_type_ids)):
                ex.context_feature[i].type_id = tokens_type_ids[i]
                ex.context_feature[i].type_prob = tokens_type_probs[i]
                ex.context_plus_question_feature[i].type_id = tokens_type_ids[i]
                ex.context_plus_question_feature[i].type_prob = tokens_type_probs[i]
        
        context_plus_question_with_types = task.create_sentence_plus_types_tokens(ex.context_plus_question,
                                                                                  ex.context_plus_question_feature,
                                                                                  self.args.add_types_to_text)
        ex = ex._replace(context_plus_question_with_types=context_plus_question_with_types)

        return ex

    def handle_request(self, line):
        if isinstance(line, dict):
            request = line
        else:
            request = json.loads(line)

        task_name = request['task'] if 'task' in request else 'generic'
        task = list(get_tasks([task_name], self.args, self._cached_task_names).values())[0]
        if task_name not in self._cached_task_names:
            self._cached_task_names[task_name] = task
            
        # if single example wrap it as a list
        if 'instances' not in request:
            request['instances'] = [{'example_id': request.get('example_id', ''), 'context': request['context'], 'question': request['question'], 'answer': request.get('answer', '')}]
        
        examples = []
        # request['instances'] is an array of {context, question, answer, example_id}
        for instance in request['instances']:
            example_id, context, question, answer = instance.get('example_id', ''), instance['context'], instance['question'], instance.get('answer', '')
            if not context:
                context = task.default_context
            if not question:
                question = task.default_question

            ex = Example.from_raw(str(example_id), context, question, answer, preprocess=task.preprocess_field, lower=self.args.lower)
            examples.append(ex)
        
        with torch.no_grad():
            bootleg_inputs = []
            if self.bootleg_annotator:
                for ex in examples:
                    if task.is_contextual():
                        bootleg_inputs.append(ex.question)
                    else:
                        bootleg_inputs.append(ex.context)
    
                bootleg_labels = self.bootleg_annotator.label_mentions(bootleg_inputs)
                bootleg_labels_unpacked = list(zip(*bootleg_labels))
                
                for i in range(len(examples)):
                    ex = examples[i]
                    label = bootleg_labels_unpacked[i]
                    examples[i] = self.bootleg_process_examples(ex, label, task)
    
            self.model.add_new_vocab_from_data([task])
            batch = self.numericalize_examples(examples)
            # it is a single batch, so wrap it in []
            if self.args.calibrator_path is not None:
                output = generate_with_model(self.model, [batch], self.numericalizer, task, self.args,
                                                  output_predictions_only=True,
                                                  confidence_estimator=self.confidence_estimator)
    
                response = json.dumps({'id': request['id'], 'instances': [{'answer': p[0], 'score': float(s)}
                                                                          for (p, s) in zip(output.predictions, output.confidence_scores)]})
            else:
                output = generate_with_model(self.model, [batch], self.numericalizer, task, self.args, output_predictions_only=True)
    
                response = json.dumps({'id': request['id'], 'instances': [{'answer': p[0]} for p in output.predictions]})
        
        return response + '\n'


    async def handle_client(self, client_reader, client_writer):
        try:
            line = await client_reader.readline()
            while line:
                client_writer.write(self.handle_request(line).encode('utf-8'))
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
                sys.stdout.write(self.handle_request(line))
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


def parse_argv(parser):
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--devices', default=[0], nargs='+', type=int,
                        help='a list of devices that can be used (multi-gpu currently WIP)')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--embeddings', default='.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--checkpoint_name', default='best.pth',
                        help='Checkpoint file to use (relative to --path, defaults to best.pth)')
    parser.add_argument('--port', default=8401, type=int, help='TCP port to listen on')
    parser.add_argument('--stdin', action='store_true', help='Interact on stdin/stdout instead of TCP')
    parser.add_argument('--database_dir', type=str, help='Database folder containing all relevant files')
    parser.add_argument('--locale', default='en', help='locale tag of the language to parse')
    parser.add_argument('--inference_name', default='nlp', help='name used by kfserving inference service, alphanumeric only')

    # for confidence estimation:
    parser.add_argument('--calibrator_path', type=str, default=None,
                        help='If provided, will be used to output confidence scores for each prediction. Defaults to `--path`/calibrator.pkl')

def init(args):
    load_config_json(args)
    set_seed(args)
    
    devices = init_devices(args)
    device = devices[0] # server only runs on a single device

    bootleg_annotator = None
    if args.do_ned and args.ned_retrieve_method == 'bootleg':
        # instantiate a bootleg object to load config and relevant databases
        bootleg = Bootleg(args)
        bootleg_config = bootleg.create_config(bootleg.fixed_overrides)

        # instantiate the annotator class. we use annotator only in server mode
        # for training we use bootleg functions which preprocess and cache data using multiprocessing, and batching to speed up NED
        bootleg_annotator = Annotator(config_args=bootleg_config,
                                      device='cpu' if device.type=='cpu' else 'cuda',
                                      max_alias_len=args.max_entity_len,
                                      cand_map=bootleg.cand_map,
                                      threshold=args.bootleg_prob_threshold)
        # collect all outputs now; we will filter later
        bootleg_annotator.set_threshold(0.0)
        setattr(bootleg_annotator, 'bootleg', bootleg)


    logger.info(f'Arguments:\n{pformat(vars(args))}')
    logger.info(f'Loading from {args.best_checkpoint}')

    Model = getattr(models, args.model)
    model, _ = Model.from_pretrained(args.path,
                                     model_checkpoint_file=args.checkpoint_name,
                                     args=args,
                                     device=device,
                                     locale=args.locale
                                     )

    model.to(device)
    model.eval()

    # set the default path for calibrator if it exists
    if args.calibrator_path is None:
        default_path = os.path.join(args.path, 'calibrator.pkl')
        if os.path.isfile(default_path):
            args.calibrator_path = default_path

    confidence_estimator = None
    if args.calibrator_path is not None:
        confidence_estimator = ConfidenceEstimator.load(args.calibrator_path)
        logger.info('Loading confidence estimator "%s" from %s', confidence_estimator.name, args.calibrator_path)
        args.mc_dropout = confidence_estimator.mc_dropout
        args.mc_dropout_num = confidence_estimator.mc_dropout_num
    return model, device, confidence_estimator, bootleg_annotator


def main(args):
    model, device, confidence_estimator, bootleg_annotator = init(args)
    server = Server(args, model.numericalizer, model, device, confidence_estimator, bootleg_annotator)
    server.run()
