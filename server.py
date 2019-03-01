#!/usr/bin/env python3
import os
from text import torchtext
from argparse import ArgumentParser
import ujson as json
import torch
import numpy as np
import random
import asyncio
import logging
from copy import deepcopy
from pprint import pformat

from util import set_seed
import models

from text.torchtext.data import Example
from text.torchtext.datasets.generic import CONTEXT_SPECIAL, QUESTION_SPECIAL, get_context_question, CQA

logger = logging.getLogger(__name__)

class ProcessedExample():
    pass


def split_tokenize(x):
    return x.split()


class Server():
    def __init__(self, args, field, model):
        self.device = set_seed(args)
        self.args = args
        self.field = field
        self.model = model

    def prepare_data(self):
        print(f'Vocabulary has {len(self.field.vocab)} tokens from training')

        char_vectors = torchtext.vocab.CharNGram(cache=args.embeddings)
        glove_vectors = torchtext.vocab.GloVe(cache=args.embeddings)
        vectors = [char_vectors, glove_vectors]
        self.field.vocab.load_vectors(vectors, True)
        self.field.decoder_to_vocab = {idx: self.field.vocab.stoi[word] for idx, word in enumerate(self.field.decoder_itos)}
        self.field.vocab_to_decoder = {idx: self.field.decoder_stoi[word] for idx, word in enumerate(self.field.vocab.itos) if word in self.field.decoder_stoi}
        
        self._limited_idx_to_full_idx = deepcopy(field.decoder_to_vocab) # should avoid this with a conditional in map to full
        self._oov_to_limited_idx = {}
        
        assert self.field.include_lengths

    def numericalize_example(self, ex):
        processed = ProcessedExample()
        
        for name in CQA.fields:
            # batch of size 1
            batch = [getattr(ex, name)]
            entry, lengths, limited_entry, raw = field.process(batch, device=self.device, train=True, 
                limited=field.decoder_stoi, l2f=self._limited_idx_to_full_idx, oov2l=self._oov_to_limited_idx)
            setattr(processed, name, entry)
            setattr(processed, f'{name}_lengths', lengths)
            setattr(processed, f'{name}_limited', limited_entry)
            setattr(processed, f'{name}_elmo', [[s.strip() for s in l] for l in raw])

        processed.oov_to_limited_idx = self._oov_to_limited_idx
        processed.limited_idx_to_full_idx = self._limited_idx_to_full_idx
        return processed

    async def handle_client(self, client_reader, client_writer):
        try:
            request = json.loads(await client_reader.readline())
            
            task = request['task'] if 'task' in request else 'generic'
            
            context = request['context']
            question = request['question']
            answer = ''
            if task == 'almond':
                tokenize = split_tokenize
            else:
                tokenize = None
        
            context_question = get_context_question(context, question)
            fields = [(x, field) for x in CQA.fields]
            ex = Example.fromlist([context, question, answer, CONTEXT_SPECIAL, QUESTION_SPECIAL, context_question], fields, tokenize=tokenize)
            
            batch = self.numericalize_example(ex)
            _, prediction_batch = model(batch, iteration=0)
            
            if task == 'almond':
                predictions = field.reverse(prediction_batch, detokenize=lambda x: ' '.join(x))
            else:
                predictions = field.reverse(prediction_batch)
                
            client_writer.write((json.dumps(dict(id=request['id'], answer=predictions[0])) + '\n').encode('utf-8'))
    
        except IOError:
            logger.info('Connection to client_reader closed')
            try:
                client_writer.close()
            except IOError:
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
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        num_param = mult(params)
        print(f'{args.model} has {num_param:,} parameters')
        model.to(self.device)
    
        model.eval()
        with torch.no_grad():
            loop = asyncio.get_event_loop()
            server = loop.run_until_complete(asyncio.start_server(self.handle_client, port=args.port))
            try:
                loop.run_forever()
            except KeyboardInterrupt:
                pass
            server.close()
            loop.run_until_complete(server.wait_closed())
            loop.close()


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used (multi-gpu currently WIP)')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--data', default='./decaNLP/.data/', type=str, help='where to load data from.')
    parser.add_argument('--embeddings', default='./decaNLP/.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--checkpoint_name', default='best.pth', help='Checkpoint file to use (relative to --path, defaults to best.pth)')
    parser.add_argument('--port', default=8401, type=int, help='TCP port to listen on')

    args = parser.parse_args()

    with open(os.path.join(args.path, 'config.json')) as config_file:
        config = json.load(config_file)
        retrieve = ['model',
                    'transformer_layers', 'rnn_layers', 'transformer_hidden', 
                    'dimension', 'load', 'max_val_context_length', 'val_batch_size', 
                    'transformer_heads', 'max_output_length', 'max_generative_vocab', 
                    'lower', 'cove', 'intermediate_cove', 'elmo', 'glove_and_char', 'use_maxmargin_loss',
                    'reverse_task_bool']
        for r in retrieve:
            if r in config:
                setattr(args, r,  config[r])
            elif 'cove' in r:
                setattr(args, r, False)
            elif 'elmo' in r:
                setattr(args, r, [-1])
            elif 'glove_and_char' in r:
                setattr(args, r, True)
            else:
                setattr(args, r, None)
        args.dropout_ratio = 0.0

    args.task_to_metric = {
        'cnn_dailymail': 'avg_rouge',
        'iwslt.en.de': 'bleu',
        'multinli.in.out': 'em',
        'squad': 'nf1',
        'srl': 'nf1',
        'almond': 'bleu' if args.reverse_task_bool else 'em',
        'sst': 'em',
        'wikisql': 'lfem',
        'woz.en': 'joint_goal_em',
        'zre': 'corpus_f1',
        'schema': 'em'
    }

    args.best_checkpoint = os.path.join(args.path, args.checkpoint_name)
    return args


if __name__ == '__main__':
    args = get_args()
    print(f'Arguments:\n{pformat(vars(args))}')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f'Loading from {args.best_checkpoint}')

    if torch.cuda.is_available():
        save_dict = torch.load(args.best_checkpoint)
    else:
        save_dict = torch.load(args.best_checkpoint, map_location='cpu')

    field = save_dict['field']
    print(f'Initializing Model')
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
    server.prepare_data()
    model.set_embeddings(field.vocab.vectors)

    server.run()
