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
from collections import OrderedDict

import torch

from .data_utils.progbar import progress_bar
from .metrics import compute_metrics
from .models import TransformerForSequenceClassification, TransformerForTokenClassification
from .util import GenerationOutput, merge_translated_sentences


def generate_with_model(
    model,
    data_iterator,
    numericalizer,
    task,
    args,
    output_predictions_only=False,
    output_confidence_features=False,
    original_order=None,
    confidence_estimators=None,
    disable_progbar=True,
):

    if isinstance(model, TransformerForTokenClassification) or isinstance(model, TransformerForSequenceClassification):
        return generate_with_classification_model(
            model, data_iterator, numericalizer, task, original_order=original_order, disable_progbar=disable_progbar
        )
    else:
        return generate_with_seq2seq_model(
            model,
            data_iterator,
            numericalizer,
            task,
            args,
            output_predictions_only=output_predictions_only,
            output_confidence_features=output_confidence_features,
            original_order=original_order,
            confidence_estimators=confidence_estimators,
            disable_progbar=disable_progbar,
        )


def generate_with_seq2seq_model(
    model,
    data_iterator,
    numericalizer,
    task,
    args,
    output_predictions_only=False,
    output_confidence_features=False,
    original_order=None,
    confidence_estimators=None,
    disable_progbar=True,
) -> GenerationOutput:
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
    output_confidence_scores = confidence_estimators is not None
    predictions = []
    confidence_features = []
    example_ids = []
    answers = []
    contexts = []

    for batch in progress_bar(data_iterator, desc='Generating', disable=disable_progbar):
        batch_size = len(batch.example_id)
        batch_prediction = [[] for _ in range(batch_size)]
        batch_confidence_features = [[] for _ in range(batch_size)]
        batch_example_ids = batch.example_id

        example_ids += batch_example_ids
        if not output_predictions_only:
            batch_answer = numericalizer.reverse(batch.answer.value.data, 'answer')
            batch_answer = [
                task.postprocess_prediction(batch_example_ids[i], batch_answer[i]) for i in range(len(batch_answer))
            ]
            answers += batch_answer
            batch_context = numericalizer.reverse(batch.context.value.data, 'context')
            contexts += batch_context
        elif output_confidence_features:
            # need gold answer for confidence estimation
            batch_answer = numericalizer.reverse(batch.answer.value.data, 'answer')
            answers += batch_answer

        for hyperparameter_idx in range(len(args.temperature)):
            generated = model.generate(
                batch,
                max_output_length=args.max_output_length,
                num_outputs=args.num_outputs[hyperparameter_idx] if args.temperature[hyperparameter_idx] != 0 else 1,
                temperature=args.temperature[hyperparameter_idx] if args.temperature[hyperparameter_idx] > 0 else 1.0,
                repetition_penalty=args.repetition_penalty[hyperparameter_idx],
                top_k=args.top_k[hyperparameter_idx],
                top_p=args.top_p[hyperparameter_idx],
                num_beams=args.num_beams[hyperparameter_idx],
                num_beam_groups=args.num_beam_groups[hyperparameter_idx],
                diversity_penalty=args.diversity_penalty[hyperparameter_idx],
                no_repeat_ngram_size=args.no_repeat_ngram_size[hyperparameter_idx],
                do_sample=args.temperature[hyperparameter_idx] != 0,  # if temperature==0, we do not sample
            )
            partial_batch_prediction_ids = generated.sequences
            cross_attentions = getattr(generated, 'cross_attentions', None)

            if cross_attentions is not None:
                # stack tensors to shape (max_output_length, num_layers, batch_size, num_heads, 1, max_input_length)
                cross_attentions = torch.stack(([torch.stack(tuple) for tuple in cross_attentions]))

                # reshape to (num_layers, batch_size, num_heads, max_output_length, max_input_length)
                cross_attentions = cross_attentions.squeeze(4)
                cross_attentions = cross_attentions.permute(1, 2, 3, 0, 4).contiguous()

                # choose only last layer attentions
                # cross_attentions = torch.mean(cross_attentions[-3:, ...], dim=0)
                cross_attentions = cross_attentions[-1, ...]

                # postprocess prediction ids
                kwargs = {'numericalizer': numericalizer, 'cross_attentions': cross_attentions}
                partial_batch_prediction_ids = task.batch_postprocess_prediction_ids(
                    batch_example_ids, batch.context.value.data, partial_batch_prediction_ids, **kwargs
                )

            if output_confidence_features or output_confidence_scores:
                partial_batch_confidence_features = model.confidence_features(
                    batch=batch, predictions=partial_batch_prediction_ids, mc_dropout_num=args.mc_dropout_num
                )

            partial_batch_prediction = numericalizer.reverse(partial_batch_prediction_ids, 'answer')

            def get_example_index(i):
                return (i // args.num_outputs[hyperparameter_idx]) % batch_size

            # post-process predictions
            for i in range(len(partial_batch_prediction)):
                partial_batch_prediction[i] = task.postprocess_prediction(
                    batch_example_ids[get_example_index(i)], partial_batch_prediction[i]
                )

            # put them into the right array
            for i in range(len(partial_batch_prediction)):
                batch_prediction[get_example_index(i)].append(partial_batch_prediction[i])
                if output_confidence_features or output_confidence_scores:
                    batch_confidence_features[get_example_index(i)].append(partial_batch_confidence_features[i])

        predictions += batch_prediction
        confidence_features += batch_confidence_features

    if original_order is not None:
        # sort back to the original order
        original_order, example_ids, predictions, answers, contexts, confidence_features = [
            list(a)
            for a in tuple(
                zip(*sorted(list(zip(original_order, example_ids, predictions, answers, contexts, confidence_features))))
            )
        ]

    if getattr(args, 'translate_example_split', False):
        # stitch sentences back together
        example_ids, predictions, answers, contexts, confidence_features = merge_translated_sentences(
            example_ids,
            predictions,
            answers,
            contexts,
            confidence_features,
            numericalizer._tokenizer.src_lang,
            numericalizer._tokenizer.tgt_lang,
        )

    # TODO calculate and return loss
    loss = None
    output = GenerationOutput(loss=loss)

    if output_predictions_only:
        output.predictions = predictions
    else:
        output.example_ids, output.predictions, output.answers, output.contexts = example_ids, predictions, answers, contexts
    if output_confidence_features:
        output.confidence_features = confidence_features
        if args.override_confidence_labels:
            for i, example in enumerate(confidence_features):
                for confidence in example:
                    confidence.label = answers[i] == args.override_confidence_labels
    if output_confidence_scores:
        output.confidence_scores = []
        for estimator in confidence_estimators:
            confidence_scores = estimator.estimate(confidence_features)
            output.confidence_scores.append(confidence_scores)

    return output


def generate_with_classification_model(
    model, data_iterator, numericalizer, task, original_order=None, disable_progbar=True
) -> GenerationOutput:
    all_example_ids = []
    all_answers = []
    all_contexts = []
    all_predictions = []

    for batch in progress_bar(data_iterator, desc='Generating', disable=disable_progbar):
        batch_example_ids = batch.example_id

        batch_context = numericalizer.reverse(batch.context.value.data, 'context')

        all_example_ids += batch_example_ids

        output = model(input_ids=batch.context.value, attention_mask=(batch.context.value != numericalizer.pad_id))

        labels = batch.answer.value.tolist()

        logits = output.logits
        predictions = torch.argmax(logits, dim=-1).tolist()

        # logits for sequence classification is 2 dimensional
        if logits.dim() == 2:
            predictions = [[p] for p in predictions]

        # Remove ignored index (special tokens)
        processed_preds = []
        processed_labels = []
        for pred, label in zip(predictions, labels):
            preds_list = []
            labels_list = []
            for p_, l_ in zip(pred, label):
                if l_ == numericalizer.answer_pad_id:
                    continue
                preds_list.append(task.id2label[p_])
                labels_list.append(task.id2label[l_])

            processed_preds.append([" ".join(preds_list)])
            processed_labels.append(" ".join(labels_list))

        all_contexts += batch_context
        all_answers += processed_labels
        all_predictions += processed_preds

    if original_order is not None:
        # sort back to the original order
        original_order, all_example_ids, all_predictions, all_answers, all_contexts = [
            list(a)
            for a in tuple(
                zip(*sorted(list(zip(original_order, all_example_ids, all_predictions, all_answers, all_contexts))))
            )
        ]

    # TODO calculate and return loss
    loss = None
    output = GenerationOutput(
        loss=loss, example_ids=all_example_ids, contexts=all_contexts, answers=all_answers, predictions=all_predictions
    )

    return output


def calculate_and_reduce_metrics(predictions, answers, metrics_to_compute, reduce_metrics, lang):
    metrics = OrderedDict()
    for i in range(len(predictions[0])):
        partial_metrics, _ = compute_metrics([p[i] for p in predictions], answers, metrics_to_compute, lang)
        for k, v in partial_metrics.items():
            if reduce_metrics == 'max':
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
        if isinstance(model, torch.nn.DataParallel):
            # get rid of the DataParallel wrapper
            model = model.module

        names = ['beam search', 'answer', 'context']

        output = generate_with_model(model, val_iter, numericalizer, task, args)

        metrics = calculate_and_reduce_metrics(
            output.predictions, output.answers, task.metrics, args.reduce_metrics, model.tgt_lang
        )
        results = [output.predictions, output.answers, output.contexts]
        print_results(names, results, num_print=num_print)

        return output, metrics
