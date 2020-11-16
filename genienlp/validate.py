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
from collections import OrderedDict

from .metrics import compute_metrics
from .util import multiwoz_specific_postprocess


def generate_with_model(model, data_iterator, numericalizer, task, args, prediction_file_name=None, output_predictions_only=False, original_order=None):
    """
    original_order: List of indices. If provided, we will sort the results according to this order
    """
    if isinstance(model, torch.nn.DataParallel):
        # get rid of the DataParallel wrapper
        model = model.module
    predictions = []
    example_ids = []
    answers = []
    contexts = []
    questions = []
    
    for batch in data_iterator:
        batch_size = len(batch.example_id)
        batch_prediction = [[] for _ in range(batch_size)] # a list where each element is a list of outputs for one input
        for hyperparameter_idx in range(len(args.temperature)):
            partial_batch_prediction = model.generate(batch,
                                                max_output_length=args.max_output_length,
                                                num_outputs=args.num_outputs[hyperparameter_idx],
                                                temperature=args.temperature[hyperparameter_idx] if args.temperature[hyperparameter_idx] > 0 else 1.0,
                                                repetition_penalty=args.repetition_penalty[hyperparameter_idx],
                                                top_k=args.top_k[hyperparameter_idx],
                                                top_p=args.top_p[hyperparameter_idx],
                                                num_beams=args.num_beams[hyperparameter_idx],
                                                no_repeat_ngram_size=args.no_repeat_ngram_size[hyperparameter_idx],
                                                do_sample=args.temperature[hyperparameter_idx]!=0  # if temperature==0, we do not sample
                                                )
            partial_batch_prediction = numericalizer.reverse(partial_batch_prediction, detokenize=task.detokenize, field_name='answer')
            if args.almond_dataset_specific_preprocess == 'multiwoz':
                partial_batch_prediction = [multiwoz_specific_postprocess(a) for a in partial_batch_prediction]
            for i in range(len(partial_batch_prediction)):
                batch_prediction[(i//args.num_outputs[hyperparameter_idx]) % batch_size].append(partial_batch_prediction[i])
        
        if not output_predictions_only:
            batch_answer = numericalizer.reverse(batch.answer.value.data, detokenize=task.detokenize, field_name='answer')
            if args.almond_dataset_specific_preprocess == 'multiwoz':
                batch_answer = [multiwoz_specific_postprocess(a) for a in batch_answer]
            example_ids += batch.example_id
            answers += batch_answer
            batch_question = numericalizer.reverse(batch.question.value.data, detokenize=task.detokenize, field_name='question')
            questions += batch_question
            batch_context = numericalizer.reverse(batch.context.value.data, detokenize=task.detokenize, field_name='context')
            contexts += batch_context
        predictions += batch_prediction
    
    if original_order is not None:
        # sort back to the original order
        original_order, example_ids, predictions, answers, contexts, questions = tuple(zip(*sorted(list(zip(original_order, example_ids, predictions, answers, contexts, questions)))))

    if prediction_file_name is not None:
        with open(prediction_file_name, 'w' + ('' if args.overwrite else 'x')) as prediction_file:
            for i in range(len(example_ids)):
                prediction_file.write(example_ids[i] + '\t' + '\t'.join(predictions[i]) + '\n') # write all outputs in the prediction file, separated by \t
    
    if output_predictions_only:
        return predictions
    # TODO calculate and return loss
    loss = None
    return loss, predictions, answers, contexts, questions

def calculate_and_reduce_metrics(predictions, answers, metrics_to_compute, args):
    metrics = OrderedDict()
    for i in range(len(predictions[0])):
        partial_metrics, _ = compute_metrics([p[i] for p in predictions], answers, metrics_to_compute)
        for k, v in partial_metrics.items():
            if args.reduce_metrics == 'max':
                metrics[k] = max(metrics.get(k, 0), v)
            else:
                raise ValueError('Invalid reduce_metrics argument')
    return metrics

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


def validate(task, val_iter, model, numericalizer, args, num_print=10):
    with torch.no_grad():
        model.eval()
        names = ['beam search', 'answer', 'context', 'question']
        loss, predictions, answers, contexts, questions = generate_with_model(model, val_iter, numericalizer, task, args, prediction_file_name=None)

        # predictions is a list of lists
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                predictions[i][j] = predictions[i][j].replace('UNK', 'OOV')

        metrics = calculate_and_reduce_metrics(predictions, answers, task.metrics, args)
        results = [predictions, answers, contexts, questions]
        print_results(names, results, num_print=num_print)

        return loss, metrics
