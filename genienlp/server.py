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
from .data_utils.embeddings import load_embeddings
from .data_utils.example import Batch
from .tasks.generic_dataset import Example
from .tasks.registry import get_tasks
from .util import set_seed, init_devices, load_config_json, log_model_size

logger = logging.getLogger(__name__)


class Server:
    def __init__(self, args, numericalizer, embeddings, model, device):
        self.args = args
        self.device = device
        self.numericalizer = numericalizer
        self.model = model

        logger.info(f'Vocabulary has {numericalizer.num_tokens} tokens from training')
        self._embeddings = embeddings

        self._cached_tasks = dict()

    def numericalize_example(self, ex):
        new_words = self.numericalizer.grow_vocab([ex])
        for emb in self._embeddings:
            emb.grow_for_vocab(self.numericalizer.vocab, new_words)

        # batch of size 1
        return Batch.from_examples([ex], self.numericalizer, device=self.device)

    def handle_request(self, line):
        request = json.loads(line)

        task_name = request['task'] if 'task' in request else 'generic'
        if task_name in self._cached_tasks:
            task = self._cached_tasks[task_name]
        else:
            task = get_tasks([task_name], self.args)[0]
            self._cached_tasks[task_name] = task

        context = request['context']
        if not context:
            context = task.default_context
        question = request['question']
        if not question:
            question = task.default_question
        answer = ''

        ex = Example.from_raw(str(request['id']), context, question, answer, tokenize=task.tokenize,
                              lower=self.args.lower)

        batch = self.numericalize_example(ex)
        _, prediction_batch = self.model(batch, iteration=0)
        predictions = self.numericalizer.reverse(prediction_batch, detokenize=task.detokenize, field_name='answer')

        response = json.dumps(dict(id=request['id'], answer=predictions[0]))
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
    save_dict = torch.load(args.best_checkpoint, map_location=devices[0])

    numericalizer, context_embeddings, question_embeddings, decoder_embeddings = \
        load_embeddings(args.embeddings, args.context_embeddings, args.question_embeddings,
                        args.decoder_embeddings, args.max_generative_vocab)
    numericalizer.load(args.path)
    for emb in set(context_embeddings + question_embeddings + decoder_embeddings):
        emb.init_for_vocab(numericalizer.vocab)

    logger.info(f'Initializing Model')
    Model = getattr(models, args.model)
    model = Model(numericalizer, args, context_embeddings, question_embeddings, decoder_embeddings)
    model_dict = save_dict['model_state_dict']
    model.load_state_dict(model_dict)

    server = Server(args, numericalizer, context_embeddings + question_embeddings + decoder_embeddings,
                    model, devices[0])

    server.run()
