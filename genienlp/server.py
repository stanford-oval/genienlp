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
import json
import logging
import sys
from pprint import pformat

import torch

from . import models
from .data_utils.example import NumericalizedExamples
from .data_utils.numericalizer.sequential_field import SequentialField
from .tasks.generic_dataset import Example
from .tasks.registry import get_tasks
from .util import set_seed, init_devices, load_config_json, log_model_size
from .validate import generate_with_model

logger = logging.getLogger(__name__)


class Server:
    def __init__(self, args, numericalizer, model, device):
        self.args = args
        self.device = device
        self.numericalizer = numericalizer
        self.model = model

        self._cached_tasks = dict()

    def numericalize_examples(self, ex):
        self.model.add_new_vocab_from_data([[ex]])

        all_features = NumericalizedExamples.from_examples(ex, self.numericalizer, device=self.device,
                                   append_question_to_context_too=self.args.append_question_to_context_too,
                                   override_question=self.args.override_question,
                                   override_context=self.args.override_context)
        all_f = []
        for i in range(len(all_features.example_id)):
            all_f.append(NumericalizedExamples(example_id=[all_features.example_id[i]],
                                context=SequentialField(value=all_features.context.value[i], length=all_features.context.length[i], limited=all_features.context.limited[i]),
                                question=SequentialField(value=all_features.question.value[i], length=all_features.question.length[i], limited=all_features.question.limited[i]),
                                answer=SequentialField(value=all_features.answer.value[i], length=all_features.answer.length[i], limited=all_features.answer.limited[i]),
                                decoder_vocab=all_features.decoder_vocab, device=self.device, padding_function=self.numericalizer.pad))

        # batch of size 1
        return NumericalizedExamples.collate_batches(all_f)

    def handle_request(self, line):
        request = json.loads(line)

        task_name = request['task'] if 'task' in request else 'generic'
        if task_name in self._cached_tasks:
            task = self._cached_tasks[task_name]
        else:
            task = list(get_tasks([task_name], self.args).values())[0]
            self._cached_tasks[task_name] = task

        if 'instances' in request:
            examples = []
            # request['instances'] is an array of {context, question, answer, example_id}
            for instance in request['instances']:
                example_id, context, question, answer = instance.get('example_id', ''), instance['context'], instance['question'], instance.get('answer', '')
                if not context:
                    context = task.default_context
                if not question:
                    question = task.default_question

                ex = Example.from_raw(str(example_id), context, question, answer, tokenize=task.tokenize, lower=self.args.lower)
                examples.append(ex)

            batch = self.numericalize_examples(examples)
            # it is a single batch, so wrap it in []
            predictions = generate_with_model(self.model, [batch], self.numericalizer, task, self.args, prediction_file_name=None, output_predictions_only=True)

            response = json.dumps({ 'id': request['id'], 'instances': [{ 'answer': p[0] } for p in predictions] })
            return response + '\n'
        else:
            context = request['context']
            if not context:
                context = task.default_context
            question = request['question']
            if not question:
                question = task.default_question
            answer = ''

            ex = Example.from_raw(str(request['id']), context, question, answer, tokenize=task.tokenize, lower=self.args.lower)

            batch = self.numericalize_examples([ex])
            predictions = generate_with_model(self.model, [batch], self.numericalizer, task, self.args, prediction_file_name=None, output_predictions_only=True)

            response = json.dumps(dict(id=request['id'], answer=predictions[0][0]))
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
        with torch.no_grad():
            if self.args.stdin:
                self._run_stdin()
            else:
                self._run_tcp()


def parse_argv(parser):
    parser.add_argument('--path', required=True)
    parser.add_argument('--devices', default=[0], nargs='+', type=int,
                        help='a list of devices that can be used (multi-gpu currently WIP)')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--embeddings', default='.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--checkpoint_name', default='best.pth',
                        help='Checkpoint file to use (relative to --path, defaults to best.pth)')
    parser.add_argument('--port', default=8401, type=int, help='TCP port to listen on')
    parser.add_argument('--stdin', action='store_true', help='Interact on stdin/stdout instead of TCP')


def main(args):
    load_config_json(args)
    set_seed(args)

    logger.info(f'Arguments:\n{pformat(vars(args))}')
    logger.info(f'Loading from {args.best_checkpoint}')

    devices = init_devices(args)
    device = devices[0] # server only runs on a single device

    Model = getattr(models, args.model)
    model, _ = Model.from_pretrained(args.path,
                                     model_checkpoint_file=args.checkpoint_name,
                                     args=args,
                                     device=device
                                    )

    
    model.to(device)
    model.eval()

    server = Server(args, model.numericalizer, model, device)

    server.run()
