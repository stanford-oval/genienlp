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

from .metrics import compute_metrics
from .util import pad


def compute_validation_outputs(model, val_iter, numericalizer, iteration):
    loss, predictions, answers, contexts, questions = [], [], [], [], []
    for batch_idx, batch in enumerate(val_iter):
        l, p = model(batch, iteration)
        loss.append(l)
        predictions.append(pad(p, 500, dim=-1, val=numericalizer.pad_id))
        a = pad(batch.answer.value.data.cpu(), 500, dim=-1, val=numericalizer.pad_id)
        answers.append(a)
        c = pad(batch.context.value.data.cpu(), 500, dim=-1, val=numericalizer.pad_id)
        contexts.append(c)
        q = pad(batch.question.value.data.cpu(), 500, dim=-1, val=numericalizer.pad_id)
        questions.append(q)

    loss = torch.cat(loss, 0) if loss[0] is not None else None
    predictions = torch.cat(predictions, 0)
    answers = torch.cat(answers, 0)
    contexts = torch.cat(contexts, 0)
    questions = torch.cat(questions, 0)
    return loss, predictions, answers, contexts, questions


def gather_results(model, val_iter, numericalizer, task, iteration):
    loss, predictions, answers, contexts, questions = \
        compute_validation_outputs(model, val_iter, numericalizer, iteration)
    answers = numericalizer.reverse(answers, detokenize=task.detokenize, field_name='answer')
    predictions = numericalizer.reverse(predictions, detokenize=task.detokenize, field_name='answer')
    contexts = numericalizer.reverse(contexts, detokenize=task.detokenize, field_name='context')
    questions = numericalizer.reverse(questions, detokenize=task.detokenize, field_name='question')

    return loss, predictions, answers, contexts, questions


def print_results(keys, values, num_print=1):
    print()
    start = 0
    end = start + num_print
    values = [val[start:end] for val in values]
    for ex_idx in range(len(values[0])):
        for key_idx, key in enumerate(keys):
            value = values[key_idx][ex_idx]
            v = value[0] if isinstance(value, list) else value
            print(f'{key}: {repr(v)}')
        print()


def validate(task, val_iter, model, logger, numericalizer, iteration, num_print=10, args=None):
    with torch.no_grad():
        model.eval()
        names = ['beam search', 'answer', 'context', 'question']
        loss, predictions, answers, contexts, questions = \
            gather_results(model, val_iter, numericalizer, task, iteration)
        predictions = [p.replace('UNK', 'OOV') for p in predictions]

        metrics, answers = compute_metrics(predictions, answers, task.metrics, args=args)
        results = [predictions, answers, contexts, questions]
        print_results(names, results, num_print=num_print)

        return loss, metrics
