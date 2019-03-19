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


import torch
from .util import pad, tokenizer
from .metrics import compute_metrics
from .text.torchtext.data.utils import get_tokenizer


def compute_validation_outputs(model, val_iter, field, iteration, optional_names=[]):
    loss, predictions, answers = [], [], []
    outputs = [[] for _ in range(len(optional_names))]
    for batch_idx, batch in enumerate(val_iter):
        l, p = model(batch, iteration)
        loss.append(l)
        predictions.append(pad(p, 150, dim=-1, val=field.vocab.stoi['<pad>']))
        a = None
        if hasattr(batch, 'wikisql_id'):
            a = batch.wikisql_id.data.cpu()
        elif hasattr(batch, 'squad_id'):
            a = batch.squad_id.data.cpu()
        elif hasattr(batch, 'woz_id'):
            a = batch.woz_id.data.cpu()
        else:
            a = pad(batch.answer.data.cpu(), 150, dim=-1, val=field.vocab.stoi['<pad>'])
        answers.append(a)
        for opt_idx, optional_name in enumerate(optional_names):
            outputs[opt_idx].append(getattr(batch, optional_name).data.cpu()) 
    loss = torch.cat(loss, 0) if loss[0] is not None else None
    predictions = torch.cat(predictions, 0)
    answers = torch.cat(answers, 0)
    return loss, predictions, answers, [torch.cat([pad(x, 150, dim=-1, val=field.vocab.stoi['<pad>']) for x in output], 0) for output in outputs]


def get_clip(val_iter):
    return -val_iter.extra if val_iter.extra > 0 else None


def all_reverse(tensor, world_size, task, field, clip, dim=0):
    
    if world_size > 1:
        tensor = tensor.float() # tensors must be on cpu and float for all_gather
        all_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.barrier() # all_gather is experimental for gloo, found that these barriers were necessary
        torch.distributed.all_gather(all_tensors, tensor)
        torch.distributed.barrier()
        tensor = torch.cat(all_tensors, 0).long() # tensors must be long for reverse
    
    # for distributed training, dev sets are padded with extra examples so that the
    # tensors are all of a predictable size for all_gather. `[:clip]` removes those extra examples
    return field.reverse(tensor, detokenize=task.detokenize, field_name='answer')[:clip]


def gather_results(model, val_iter, field, world_size, task, iteration, optional_names=[]):
    loss, predictions, answers, outputs = compute_validation_outputs(model, val_iter, field, iteration, optional_names=optional_names)
    clip = get_clip(val_iter)
    if not hasattr(val_iter.dataset.examples[0], 'squad_id') and not hasattr(val_iter.dataset.examples[0], 'wikisql_id') and not hasattr(val_iter.dataset.examples[0], 'woz_id'):
        answers = all_reverse(answers, world_size, task, field, clip)
    return loss, all_reverse(predictions, world_size, task, field, clip), answers, [all_reverse(x, world_size, task, field, clip) for x in outputs]


def print_results(keys, values, rank=None, num_print=1):
    print()
    start = rank * num_print if rank is not None else 0
    end = start + num_print
    values = [val[start:end] for val in values]
    for ex_idx in range(len(values[0])):
        for key_idx, key in enumerate(keys):
            value = values[key_idx][ex_idx]
            v = value[0] if isinstance(value, list) else value
            print(f'{key}: {repr(v)}')
        print()


def validate(task, val_iter, model, logger, field, world_size, rank, iteration, num_print=10, args=None):
    with torch.no_grad():
        model.eval()
        required_names = ['greedy', 'answer']
        optional_names = ['context', 'question']
        loss, predictions, answers, results = gather_results(model, val_iter, field, world_size, task, iteration, optional_names=optional_names)
        predictions = [p.replace('UNK', 'OOV') for p in predictions]
        names = required_names + optional_names 
        if hasattr(val_iter.dataset.examples[0], 'wikisql_id') or hasattr(val_iter.dataset.examples[0], 'squad_id') or hasattr(val_iter.dataset.examples[0], 'woz_id'):
            answers = [val_iter.dataset.all_answers[sid] for sid in answers.tolist()]

        metrics, answers = compute_metrics(predictions, answers, task.metrics, args=args)
        results = [predictions, answers] + results
        print_results(names, results, rank=rank, num_print=num_print)

        return loss, metrics
