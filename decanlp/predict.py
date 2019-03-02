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

import os
from .utils.generic_dataset import Query
from .text import torchtext
from argparse import ArgumentParser
import ujson as json
import torch
import numpy as np
import random
import sys
from pprint import pformat

from .util import get_splits, set_seed, preprocess_examples
from .metrics import compute_metrics
from . import models

def get_all_splits(args, new_vocab):
    splits = []
    for task in args.tasks:
        print(f'Loading {task}')
        kwargs = {}
        if not 'train' in args.evaluate:
            kwargs['train'] = None
        if not 'valid' in args.evaluate:
            kwargs['validation'] = None
        if not 'test' in args.evaluate:
            kwargs['test'] = None
        s = get_splits(args, task, new_vocab, **kwargs)[0]
        preprocess_examples(args, [task], [s], new_vocab, train=False)
        splits.append(s)
    return splits


def prepare_data(args, FIELD):
    new_vocab = torchtext.data.ReversibleField(batch_first=True, init_token='<init>', eos_token='<eos>', lower=args.lower, include_lengths=True)
    splits = get_all_splits(args, new_vocab)
    new_vocab.build_vocab(*splits)
    print(f'Vocabulary has {len(FIELD.vocab)} tokens from training')
    args.max_generative_vocab = min(len(FIELD.vocab), args.max_generative_vocab)
    FIELD.append_vocab(new_vocab)
    print(f'Vocabulary has expanded to {len(FIELD.vocab)} tokens')

    char_vectors = torchtext.vocab.CharNGram(cache=args.embeddings)
    glove_vectors = torchtext.vocab.GloVe(cache=args.embeddings)
    vectors = [char_vectors, glove_vectors]
    FIELD.vocab.load_vectors(vectors, True)
    FIELD.decoder_to_vocab = {idx: FIELD.vocab.stoi[word] for idx, word in enumerate(FIELD.decoder_itos)}
    FIELD.vocab_to_decoder = {idx: FIELD.decoder_stoi[word] for idx, word in enumerate(FIELD.vocab.itos) if word in FIELD.decoder_stoi}
    splits = get_all_splits(args, FIELD)

    return FIELD, splits


def to_iter(data, bs, device):
    Iterator = torchtext.data.Iterator
    it = Iterator(data, batch_size=bs, 
       device=device, batch_size_fn=None, 
       train=False, repeat=False, sort=False,
       shuffle=False, reverse=False)

    return it


def run(args, field, val_sets, model):
    device = set_seed(args)
    print(f'Preparing iterators')
    if len(args.val_batch_size) == 1 and len(val_sets) > 1:
        args.val_batch_size *= len(val_sets)
    iters = [(name, to_iter(x, bs, device)) for name, x, bs in zip(args.tasks, val_sets, args.val_batch_size)]
 
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
    model.to(device)

    decaScore = []
    model.eval()
    with torch.no_grad():
        for task, it in iters:
            print(task)
            if args.eval_dir:
                prediction_file_name = os.path.join(args.eval_dir, os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task + '.txt'))
                answer_file_name = os.path.join(args.eval_dir, os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task + '.gold.txt'))
                results_file_name = answer_file_name.replace('gold', 'results')
            else:
                prediction_file_name = os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task + '.txt')
                answer_file_name = os.path.join(os.path.splitext(args.best_checkpoint)[0], args.evaluate, task + '.gold.txt')
                results_file_name = answer_file_name.replace('gold', 'results')
            if 'sql' in task or 'squad' in task:
                ids_file_name = answer_file_name.replace('gold', 'ids')
            if os.path.exists(prediction_file_name):
                print('** ', prediction_file_name, ' already exists -- this is where predictions are stored **')
                if args.overwrite:
                    print('**** overwriting ', prediction_file_name, ' ****')
            if os.path.exists(answer_file_name):
                print('** ', answer_file_name, ' already exists -- this is where ground truth answers are stored **')
                if args.overwrite:
                    print('**** overwriting ', answer_file_name, ' ****')
            if os.path.exists(results_file_name):
                print('** ', results_file_name, ' already exists -- this is where metrics are stored **')
                if args.overwrite:
                    print('**** overwriting ', results_file_name, ' ****')
                else:
                    with open(results_file_name) as results_file:
                        if not args.silent:
                            for l in results_file:
                                print(l)
                        metrics = json.loads(results_file.readlines()[0])
                        decaScore.append(metrics[args.task_to_metric[task]])
                    continue

            for x in [prediction_file_name, answer_file_name, results_file_name]:
                os.makedirs(os.path.dirname(x), exist_ok=True)
    
            if not os.path.exists(prediction_file_name) or args.overwrite:
                with open(prediction_file_name, 'w') as prediction_file:
                    predictions = []
                    ids = []
                    for batch_idx, batch in enumerate(it):
                        _, p = model(batch, iteration=1)

                        if task == 'almond':
                            p = field.reverse(p, detokenize=lambda x: ' '.join(x))
                        else:
                            p = field.reverse(p)

                        for i, pp in enumerate(p):
                            if 'sql' in task:
                                ids.append(int(batch.wikisql_id[i]))
                            if 'squad' in task:
                                ids.append(it.dataset.q_ids[int(batch.squad_id[i])])
                            prediction_file.write(json.dumps(pp) + '\n')
                            predictions.append(pp) 
                if 'sql' in task:
                    with open(ids_file_name, 'w') as id_file:
                        for i in ids:
                            id_file.write(json.dumps(i) + '\n')
                if 'squad' in task:
                    with open(ids_file_name, 'w') as id_file:
                        for i in ids:
                            id_file.write(i + '\n')
            else:
                with open(prediction_file_name) as prediction_file:
                    predictions = [x.strip() for x in prediction_file.readlines()] 
                if 'sql' in task or 'squad' in task:
                    with open(ids_file_name) as id_file:
                        ids = [int(x.strip()) for x in id_file.readlines()]
   
            def from_all_answers(an):
                return [it.dataset.all_answers[sid] for sid in an.tolist()] 
    
            if not os.path.exists(answer_file_name) or args.overwrite:
                with open(answer_file_name, 'w') as answer_file:
                    answers = []
                    for batch_idx, batch in enumerate(it):
                        if hasattr(batch, 'wikisql_id'):
                            a = from_all_answers(batch.wikisql_id.data.cpu())
                        elif hasattr(batch, 'squad_id'):
                            a = from_all_answers(batch.squad_id.data.cpu())
                        elif hasattr(batch, 'woz_id'):
                            a = from_all_answers(batch.woz_id.data.cpu())
                        else:
                            if task == 'almond':
                                a = field.reverse(batch.answer.data, detokenize=lambda x: ' '.join(x))
                            else:
                                a = field.reverse(batch.answer.data)
                        for aa in a:
                            answers.append(aa) 
                            answer_file.write(json.dumps(aa) + '\n')
            else:
                with open(answer_file_name) as answer_file:
                    answers = [json.loads(x.strip()) for x in answer_file.readlines()] 
    
            if len(answers) > 0:
                if not os.path.exists(results_file_name) or args.overwrite:
                    metrics, answers = compute_metrics(predictions, answers,
                                           bleu='iwslt' in task or 'multi30k' in task or 'almond' in task,
                                           dialogue='woz' in task,
                                           rouge='cnn' in task, logical_form='sql' in task, corpus_f1='zre' in task,
                                           func_accuracy='almond' in task and not args.reverse_task_bool,
                                           dev_accuracy='almond' in task and not args.reverse_task_bool,
                                           args=args)
                    with open(results_file_name, 'w') as results_file:
                        results_file.write(json.dumps(metrics) + '\n')
                else:
                    with open(results_file_name) as results_file:
                        metrics = json.loads(results_file.readlines()[0])
    
                if not args.silent:
                    for i, (p, a) in enumerate(zip(predictions, answers)):
                        print(f'Prediction {i+1}: {p}\nAnswer {i+1}: {a}\n')
                    print(metrics)
                decaScore.append(metrics[args.task_to_metric[task]])

    print(f'Evaluated Tasks:\n')
    for i, (task, _) in enumerate(iters):
        print(f'{task}: {decaScore[i]}')
    print(f'-------------------')
    print(f'DecaScore:  {sum(decaScore)}\n')
    print(f'\nSummary: | {sum(decaScore)} | {" | ".join([str(x) for x in decaScore])} |\n')


def get_args(argv):
    parser = ArgumentParser(prog=argv[0])
    parser.add_argument('--path', required=True)
    parser.add_argument('--evaluate', type=str, required=True)
    parser.add_argument('--tasks', default=['almond', 'squad', 'iwslt.en.de', 'cnn_dailymail', 'multinli.in.out', 'sst', 'srl', 'zre', 'woz.en', 'wikisql', 'schema'], nargs='+')
    parser.add_argument('--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used (multi-gpu currently WIP)')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--data', default='./decaNLP/.data/', type=str, help='where to load data from.')
    parser.add_argument('--embeddings', default='./decaNLP/.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument('--checkpoint_name', default='best.pth', help='Checkpoint file to use (relative to --path, defaults to best.pth)')
    parser.add_argument('--bleu', action='store_true', help='whether to use the bleu metric (always on for iwslt)')
    parser.add_argument('--rouge', action='store_true', help='whether to use the bleu metric (always on for cnn, dailymail, and cnn_dailymail)')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite previously written predictions')
    parser.add_argument('--silent', action='store_true', help='whether to print predictions to stdout')

    parser.add_argument('--skip_cache', action='store_true', dest='skip_cache_bool', help='whether use exisiting cached splits or generate new ones')
    parser.add_argument('--reverse_task', action='store_true', dest='reverse_task_bool', help='whether to translate english to code or the other way around')
    parser.add_argument('--eval_dir', type=str, default=None, help='use this directory to store eval results')
    parser.add_argument('--cached', default='', type=str, help='where to save cached files')

    args = parser.parse_args(argv[1:])

    with open(os.path.join(args.path, 'config.json')) as config_file:
        config = json.load(config_file)
        retrieve = ['model', 
                    'transformer_layers', 'rnn_layers', 'transformer_hidden', 
                    'dimension', 'load', 'max_val_context_length', 'val_batch_size', 
                    'transformer_heads', 'max_output_length', 'max_generative_vocab', 
                    'lower', 'cove', 'intermediate_cove', 'elmo', 'glove_and_char', 'use_maxmargin_loss']
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


def main(argv=sys.argv):
    args = get_args(argv)
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
    field, splits = prepare_data(args, field)
    model.set_embeddings(field.vocab.vectors)

    run(args, field, splits, model)

if __name__ == '__main__':
    main()
