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

import sys
import torch
from collections import OrderedDict

from .util import GenerationOutput
from .metrics import compute_metrics


def generate_with_model(model, data_iterator, numericalizer, task,args,
                        output_predictions_only=False,
                        output_confidence_features=False,
                        original_order=None,
                        confidence_estimator=None) -> GenerationOutput:
    """
    Inputs:
        original_order: List of indices. If provided, we will sort the results according to this order
        confidence_estimator: if provided, will use it to calculate and output confidence scores
    Outputs: predictions if `output_predictions_only` == True, (loss, predictions, answers, contexts) otherwise
        loss
        predictions: a List of Lists of strings
        answers
        contexts
    """
    output_confidence_scores = confidence_estimator is not None
    if isinstance(model, torch.nn.DataParallel):
        # get rid of the DataParallel wrapper
        model = model.module
    predictions = []
    confidence_features = []
    example_ids = []
    answers = []
    contexts = []

    for batch in data_iterator:
        batch_size = len(batch.example_id)
        raw_batch_prediction = [[] for _ in range(batch_size)] # a list where each element is a list of outputs for one input
        batch_prediction = [[] for _ in range(batch_size)]
        batch_confidence_features = [[] for _ in range(batch_size)]

        for hyperparameter_idx in range(len(args.temperature)):
            raw_partial_batch_prediction = model.generate(batch,
                                                max_output_length=args.max_output_length,
                                                num_outputs=args.num_outputs[hyperparameter_idx],
                                                temperature=args.temperature[hyperparameter_idx] if args.temperature[hyperparameter_idx] > 0 else 1.0,
                                                repetition_penalty=args.repetition_penalty[hyperparameter_idx],
                                                top_k=args.top_k[hyperparameter_idx],
                                                top_p=args.top_p[hyperparameter_idx],
                                                num_beams=args.num_beams[hyperparameter_idx],
                                                num_beam_groups=args.num_beam_groups[hyperparameter_idx],
                                                diversity_penalty=args.diversity_penalty[hyperparameter_idx],
                                                no_repeat_ngram_size=args.no_repeat_ngram_size[hyperparameter_idx],
                                                do_sample=args.temperature[hyperparameter_idx]!=0,  # if temperature==0, we do not sample
                                                )
            if output_confidence_features or output_confidence_scores:
                partial_batch_confidence_features =  model.confidence_features(batch=batch, predictions=raw_partial_batch_prediction, mc_dropout=args.mc_dropout, mc_dropout_num=args.mc_dropout_num)
            partial_batch_prediction = numericalizer.reverse(raw_partial_batch_prediction, task=task, field_name='answer')
            for i in range(len(partial_batch_prediction)):
                batch_prediction[(i//args.num_outputs[hyperparameter_idx]) % batch_size].append(partial_batch_prediction[i])
                raw_batch_prediction[(i//args.num_outputs[hyperparameter_idx]) % batch_size].append(raw_partial_batch_prediction[i])
                if output_confidence_features or output_confidence_scores:
                    batch_confidence_features[(i//args.num_outputs[hyperparameter_idx]) % batch_size].append(partial_batch_confidence_features[i])
        
        if not output_predictions_only:
            batch_answer = numericalizer.reverse(batch.answer.value.data, task=task, field_name='answer')
            example_ids += batch.example_id
            answers += batch_answer
            batch_context = numericalizer.reverse(batch.context.value.data, task=task, field_name='context')
            contexts += batch_context
        predictions += batch_prediction
        confidence_features += batch_confidence_features
    
    if original_order is not None:
        # sort back to the original order
        original_order, example_ids, predictions, answers, contexts, confidence_features = [list(a) for a in tuple(zip(*sorted(list(zip(original_order, example_ids, predictions, answers, contexts, confidence_features)))))]
    
    # TODO calculate and return loss
    loss = None
    output = GenerationOutput(loss=loss)

    if output_predictions_only:
        output.predictions = predictions
    else:
        output.example_ids, output.predictions, output.answers, output.contexts = example_ids, predictions, answers, contexts
    if output_confidence_features:
        output.confidence_features = confidence_features
    if output_confidence_scores:
        confidence_scores = confidence_estimator.estimate(confidence_features)
        output.confidence_scores = confidence_scores

    return output


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
            print(f'{key:>11}: {repr(v)}')
        print()
    sys.stdout.flush()


def validate(task, val_iter, model, numericalizer, args, num_print=10):
    with torch.no_grad():
        model.eval()
        names = ['beam search', 'answer', 'context']
        output = generate_with_model(model, val_iter, numericalizer, task, args)

        metrics = calculate_and_reduce_metrics(output.predictions, output.answers, task.metrics, args)
        results = [output.predictions, output.answers, output.contexts]
        print_results(names, results, num_print=num_print)

        return output.loss, metrics
