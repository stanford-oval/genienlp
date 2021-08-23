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

from .data_utils.example import NumericalizedExamples, SequentialField
from .data_utils.progbar import progress_bar
from .metrics import compute_metrics
from .models import TransformerForSequenceClassification, TransformerForTokenClassification
from .util import GenerationOutput, merge_translated_sentences


def calculate_loss(model, pred_logits, answer):
    # calculate loss
    # it's not exactly meaningful to calculate loss for generation tasks
    # cause position of prediction tokens may not align with gold tokens
    batch_size, vocab_size = pred_logits.shape[0], pred_logits.shape[2]
    answer_value = answer.value[:, 1:].contiguous()
    answer_length = answer.length - 1
    partial_batch_prediction_logits = pred_logits[:, : max(answer_length), :].contiguous()
    partial_batch_prediction_logits = torch.where(
        torch.isinf(partial_batch_prediction_logits),
        torch.tensor(1e-10, device=partial_batch_prediction_logits.device),
        partial_batch_prediction_logits,
    )
    loss = model.criterion(
        partial_batch_prediction_logits.view(-1, vocab_size),
        target=answer_value.view(-1),
        ignore_index=model.numericalizer.pad_id,
    )
    loss = loss.view(batch_size, -1)  # (batch_size, sequence_length)
    loss = loss.sum(dim=1) / answer_length  # accounts for the case where BOS is removed
    if model.dropper is not None:
        dropper_mask = model.dropper(loss)
        loss = loss * dropper_mask
    loss = loss.sum()  # sum over the batch size

    return loss


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
    if args.bitod_e2e_evaluation:
        return generate_with_seq2seq_model_for_dialogue(
            model,
            data_iterator,
            numericalizer,
            task,
            args,
            output_predictions_only=output_predictions_only,
            original_order=original_order,
            disable_progbar=disable_progbar,
        )

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


def generate_with_seq2seq_model_for_dialogue(
    model,
    data_iterator,
    numericalizer,
    task,
    args,
    output_predictions_only=False,
    original_order=None,
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
    predictions = []
    example_ids = []
    answers = []
    contexts = []

    cur_dial_id = ''

    device = model.device

    for k, turn in enumerate(progress_bar(data_iterator, desc='Generating', disable=disable_progbar)):
        batch_size = len(turn.example_id)
        assert batch_size == 1
        batch_prediction = []
        batch_example_ids = turn.example_id

        example_ids += batch_example_ids

        task_name, dial_id, turn_id, train_target = batch_example_ids[0].split('/')
        if cur_dial_id != dial_id:
            # new dialogue
            cur_dial_id = dial_id
            first_turn = True
        else:
            first_turn = False

        special_tokens = numericalizer._tokenizer.all_special_tokens
        batch_tokens = numericalizer.convert_ids_to_tokens(turn.context.value.data, skip_special_tokens=False)
        batch_context = []
        # remove only beginning and trailing special tokens
        # otherwise the numericalizer.sep_token added between context and question will be lost
        for text in batch_tokens:
            i = 0
            while text[i] in special_tokens:
                i += 1
            j = len(text) - 1
            while text[j] in special_tokens:
                j -= 1
            text = text[i : j + 1]

            batch_context.append(numericalizer._tokenizer.convert_tokens_to_string(text))

        contexts += batch_context
        batch_context = numericalizer.reverse(turn.context.value.data, 'context')
        contexts += batch_context

        if not output_predictions_only:
            batch_answer = numericalizer.reverse(turn.answer.value.data, 'answer')
            batch_answer = [
                task.postprocess_prediction(batch_example_ids[i], batch_answer[i]) for i in range(len(batch_answer))
            ]
            answers += batch_answer

        # iterate through turns
        hyperparameter_idx = 0

        if first_turn:
            numericalized_turn = NumericalizedExamples(
                example_id=[turn.example_id[k]],
                context=SequentialField(
                    value=turn.context.value[[k]],
                    length=turn.context.length[[k]],
                    limited=turn.context.limited[[k]],
                    feature=None,
                ),
                answer=SequentialField(
                    value=turn.answer.value[[k]],
                    length=turn.answer.value[[k]],
                    limited=turn.answer.value[[k]],
                    feature=None,
                ),
            )
        else:
            semi_colon_idx = batch_context[k].index(';')
            context = batch_prediction[-1][0].strip() + ' ' + batch_context[k][semi_colon_idx:].strip()
            if context.startswith('$dialogue '):
                context = context[len('$dialogue ') :]
            tokenized_contexts = numericalizer.encode_batch([context], field_name='context', features=None)[0]
            value = torch.tensor([tokenized_contexts.value], device=device)
            length = torch.tensor([tokenized_contexts.length], device=device)
            limited = torch.tensor([tokenized_contexts.limited], device=device)
            numericalized_turn = NumericalizedExamples(
                example_id=[turn.example_id[k]],
                context=SequentialField(value=value, length=length, limited=limited, feature=None),
                answer=SequentialField(
                    value=turn.answer.value[[k]],
                    length=turn.answer.value[[k]],
                    limited=turn.answer.value[[k]],
                    feature=None,
                ),
            )

        generated = model.generate(
            numericalized_turn,
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
            do_sample=args.temperature[hyperparameter_idx] != 0,
        )
        partial_batch_prediction_ids = generated.sequences

        partial_batch_prediction = numericalizer.reverse(partial_batch_prediction_ids, 'answer')[0]

        # post-process predictions
        partial_batch_prediction = task.postprocess_prediction(batch_example_ids[k], partial_batch_prediction)

        # put them into the right array
        batch_prediction.append([partial_batch_prediction])

        predictions += batch_prediction

    if original_order is not None:
        # sort back to the original order
        original_order, example_ids, predictions, answers, contexts = [
            list(a) for a in tuple(zip(*sorted(list(zip(original_order, example_ids, predictions, answers, contexts)))))
        ]

    # TODO calculate and return loss
    loss = None
    output = GenerationOutput(loss=loss)

    if output_predictions_only:
        output.predictions = predictions
    else:
        output.example_ids, output.predictions, output.answers, output.contexts = example_ids, predictions, answers, contexts

    return output


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
    total_loss = 0.0 if model._output_scores else None
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

        if total_loss is not None:
            loss = model(batch, train=True).loss.item()
            # loss = calculate_loss(model, partial_batch_prediction_logits, batch.answer)
            total_loss += loss

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
            # partial_batch_prediction_logits = torch.stack(generated.scores).permute(1, 0, 2).contiguous()

            if model._output_attentions:
                cross_attentions = generated.cross_attentions

                # stack tensors to shape (max_output_length, num_layers, batch_size, num_heads, 1, max_input_length)
                cross_attentions = torch.stack(([torch.stack(tuple) for tuple in cross_attentions])).cpu()

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

    total_loss /= len(example_ids)

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

    output = GenerationOutput(loss=total_loss)

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
    total_loss = 0.0
    all_example_ids = []
    all_answers = []
    all_contexts = []
    all_predictions = []

    for batch in progress_bar(data_iterator, desc='Generating', disable=disable_progbar):
        batch_example_ids = batch.example_id

        batch_context = numericalizer.reverse(batch.context.value.data, 'context')

        all_example_ids += batch_example_ids

        # pass labels to get loss
        output = model(
            input_ids=batch.context.value,
            attention_mask=(batch.context.value != numericalizer.pad_id),
            labels=batch.answer.value,
        )

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

        # loss = calculate_loss(model, logits, batch.answer)
        total_loss += output.loss

    total_loss /= len(all_example_ids)

    if original_order is not None:
        # sort back to the original order
        original_order, all_example_ids, all_predictions, all_answers, all_contexts = [
            list(a)
            for a in tuple(
                zip(*sorted(list(zip(original_order, all_example_ids, all_predictions, all_answers, all_contexts))))
            )
        ]

    output = GenerationOutput(
        loss=total_loss, example_ids=all_example_ids, contexts=all_contexts, answers=all_answers, predictions=all_predictions
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

        validation_outputs = output
        if task.name == 'bitod' and args.bitod_validation_task != 'all':
            validation_outputs = GenerationOutput()
            for i in range(len(output.example_ids)):
                id_, train_task = output.example_ids[i].rsplit('/', 1)
                if train_task in args.bitod_validation_task:
                    validation_outputs.answers.append(output.answers[i])
                    validation_outputs.predictions.append(output.predictions[i])

        # loss is already calculated
        metrics_to_return = [metric for metric in task.metrics if metric != 'loss']

        metrics = calculate_and_reduce_metrics(
            validation_outputs.predictions, validation_outputs.answers, metrics_to_return, args.reduce_metrics, model.tgt_lang
        )

        results = [output.predictions, output.answers, output.contexts]
        print_results(names, results, num_print=num_print)

        return output, metrics
