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


from argparse import ArgumentParser
import ujson as json
import torch
import numpy as np
import random
import asyncio
import logging
import sys
from copy import deepcopy
from pprint import pformat

from .util import set_seed, load_config_json
from . import models
from .text.torchtext.data import Example
from .utils.embeddings import load_embeddings
from .tasks.registry import get_tasks
from .tasks.generic_dataset import CONTEXT_SPECIAL, QUESTION_SPECIAL, get_context_question, CQA

logger = logging.getLogger(__name__)

class ProcessedExample():
    pass

class Server():
    def __init__(self, args, field, model):
        self.device = set_seed(args)
        self.args = args
        self.field = field
        self.model = model

        logger.info(f'Vocabulary has {len(self.field.vocab)} tokens from training')
        self._vector_collections = load_embeddings(args)
        
        self._limited_idx_to_full_idx = deepcopy(self.field.decoder_to_vocab) # should avoid this with a conditional in map to full
        self._oov_to_limited_idx = {}

        self._cached_tasks = dict()
        
        assert self.field.include_lengths

    def numericalize_example(self, ex):
        processed = ProcessedExample()
        
        new_vectors = []
        for name in CQA.fields:
            value = getattr(ex, name)
            
            assert isinstance(value, list)
            # check if all the words are in the vocabulary, and if not
            # grow the vocabulary and the embedding matrix
            for word in value:
                if word not in self.field.vocab.stoi:
                    self.field.vocab.stoi[word] = len(self.field.vocab.itos)
                    self.field.vocab.itos.append(word)
                    
                    new_vector = [vec[word] for vec in self._vector_collections]
                    
                    # charNgram returns  a [1, D] tensor, while Glove returns a [D] tensor
                    # normalize to [1, D] so we can concat along the second dimension
                    # and later concat all vectors along the first
                    new_vector = [vec if vec.dim() > 1 else vec.unsqueeze(0) for vec in new_vector]
                    new_vectors.append(torch.cat(new_vector, dim=1))
            
            # batch of size 1
            batch = [value]            
            entry, lengths, limited_entry, raw = self.field.process(batch, device=self.device, train=True, 
                limited=self.field.decoder_stoi, l2f=self._limited_idx_to_full_idx, oov2l=self._oov_to_limited_idx)
            setattr(processed, name, entry)
            setattr(processed, f'{name}_lengths', lengths)
            setattr(processed, f'{name}_limited', limited_entry)
            setattr(processed, f'{name}_elmo', [[s.strip() for s in l] for l in raw])

        processed.oov_to_limited_idx = self._oov_to_limited_idx
        processed.limited_idx_to_full_idx = self._limited_idx_to_full_idx
        
        if new_vectors:
            # concat the old embedding matrix and all the new vector along the first dimension
            new_embedding_matrix = torch.cat([self.field.vocab.vectors] + new_vectors, dim=0)
            self.field.vocab.vectors = new_embedding_matrix
            self.model.set_embeddings(new_embedding_matrix)
            
        return processed
    
    def handle_request(self, line):
        request = json.loads(line)

        task_name = request['task'] if 'task' in request else 'generic'
        if task_name in self._cached_tasks:
            task = self._cached_tasks[task_name]
        else:
            task = get_tasks([task_name], self.args)[0]
            self._cached_tasks[task_name] = task
        
        context = request['context']
        question = request['question']
        answer = ''
        tokenize = task.tokenize
    
        context_question = get_context_question(context, question)
        fields = [(x, self.field) for x in CQA.fields]
        ex = Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields, tokenize=tokenize)
        
        batch = self.numericalize_example(ex)
        _, prediction_batch = self.model(batch, iteration=0)
        
        predictions = self.field.reverse(prediction_batch, detokenize=task.detokenize)
        
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
        def mult(ps):
            r = 0
            for p in ps:
                this_r = 1
                for s in p.size():
                    this_r *= s
                r += this_r
            return r
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        num_param = mult(params)
        logger.info(f'{self.args.model} has {num_param:,} parameters')
        self.model.to(self.device)
    
        self.model.eval()
        with torch.no_grad():
            if self.args.stdin:
                self._run_stdin()
            else:
                self._run_tcp()


def get_args(argv):
    parser = ArgumentParser(prog=argv[0])
    parser.add_argument('--path', required=True)
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used (multi-gpu currently WIP)')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--embeddings', default='./decaNLP/.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--thingpedia', type=str, help='where to load thingpedia.json from (for almond task only)')
    parser.add_argument('--checkpoint_name', default='best.pth', help='Checkpoint file to use (relative to --path, defaults to best.pth)')
    parser.add_argument('--port', default=8401, type=int, help='TCP port to listen on')
    parser.add_argument('--stdin', action='store_true', help='Interact on stdin/stdout instead of TCP')

    args = parser.parse_args(argv[1:])
    load_config_json(args)
    return args


def main(argv=sys.argv):
    args = get_args(argv)
    logger.info(f'Arguments:\n{pformat(vars(args))}')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info(f'Loading from {args.best_checkpoint}')

    if torch.cuda.is_available():
        save_dict = torch.load(args.best_checkpoint)
    else:
        save_dict = torch.load(args.best_checkpoint, map_location='cpu')

    field = save_dict['field']
    logger.info(f'Initializing Model')
    Model = getattr(models, args.model)
    model = Model(field, args)
    model_dict = save_dict['model_state_dict']
    backwards_compatible_cove_dict = {}
    for k, v in model_dict.items():
        if 'cove.rnn.' in k:
            k = k.replace('cove.rnn.', 'cove.rnn1.')
        backwards_compatible_cove_dict[k] = v
    model_dict = backwards_compatible_cove_dict
    model.load_state_dict(model_dict)
    
    server = Server(args, field, model)
    model.set_embeddings(field.vocab.vectors)

    server.run()
