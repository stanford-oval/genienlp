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
import os
import torch
from collections import OrderedDict

from .paraphrase.model_utils import compute_attention, replace_quoted_params, force_replace_quoted_params
from .util import GenerationOutput
from .data_utils.progbar import progress_bar
from .metrics import compute_metrics


def generate_with_model(model, data_iterator, numericalizer, task, args,
                        output_predictions_only=False,
                        output_confidence_features=False,
                        original_order=None,
                        confidence_estimators=None,
                        disable_progbar=True) -> GenerationOutput:
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
    if isinstance(model, torch.nn.DataParallel):
        # get rid of the DataParallel wrapper
        model = model.module
    predictions = []
    confidence_features = []
    example_ids = []
    answers = []
    contexts = []
    
    for j, batch in enumerate(progress_bar(data_iterator, desc='Generating', disable=disable_progbar)):
        batch_size = len(batch.example_id)
        batch_prediction = [[] for _ in range(batch_size)]
        batch_confidence_features = [[] for _ in range(batch_size)]
        batch_example_ids = batch.example_id

        example_ids += batch_example_ids
        if not output_predictions_only:
            batch_answer = numericalizer.reverse(batch.answer.value.data, 'answer')
            batch_answer = [task.postprocess_prediction(batch_example_ids[i], batch_answer[i]) for i in range(len(batch_answer))]
            answers += batch_answer
            batch_context = numericalizer.reverse(batch.context.value.data, 'context')
            contexts += batch_context
        elif output_confidence_features:
            # need gold answer for confidence estimation
            batch_answer = numericalizer.reverse(batch.answer.value.data, 'answer')
            answers += batch_answer

        for hyperparameter_idx in range(len(args.temperature)):
            generated = model.generate(batch,
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
            partial_batch_prediction_ids = generated.sequences
            
            #
            cross_attentions = generated.cross_attentions # Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of shape (batch_size, num_heads, generated_length, sequence_length)
            
            # stack tensors to shape (max_output_length, num_layers, batch_size, num_heads, 1, max_input_length)
            cross_attentions = torch.stack(([torch.stack(tuple) for tuple in cross_attentions]))
            
            # reshape to (num_layers, batch_size, num_heads, max_output_length, max_input_length)
            cross_attentions = cross_attentions.squeeze(4)
            cross_attentions = cross_attentions.permute(1, 2, 3, 0, 4).contiguous()
            
            # choose only last layer attentions
            # TODO: get penultimate layer of attention vectors instead
            cross_attentions = cross_attentions[-1, ...]

            all_src_tokens = numericalizer.convert_ids_to_tokens(batch.context.value.data, skip_special_tokens=False)
            all_tgt_tokens = numericalizer.convert_ids_to_tokens(partial_batch_prediction_ids, skip_special_tokens=False)
            
            # remove language code from the beginning of src_tokens and shift layer_attention
            len_prefix_wp = len(numericalizer._tokenizer.tokenize(numericalizer.input_prefix))
            all_src_tokens = [tokens[len_prefix_wp:] for tokens in all_src_tokens]
            cross_attentions = cross_attentions[:, :, :, len_prefix_wp:]

            cross_attention_pooled = compute_attention(cross_attentions, att_pooling=args.att_pooling, dim=1)
            
            all_tgt_strings = []
            for i, (src_tokens, tgt_tokens, cross_att) in enumerate(zip(all_src_tokens, all_tgt_tokens, cross_attention_pooled)):
                
                # shift target tokens left to match the attention positions
                if tgt_tokens[0] in numericalizer._tokenizer.all_special_tokens:
                    tgt_tokens = tgt_tokens[1:]
    
                # remove all trailing special tokens from source
                while src_tokens[-1] in numericalizer._tokenizer.all_special_tokens:
                    src_tokens = src_tokens[:-1]
    
                # crop to match src and tgt new lengths
                cross_att = cross_att[:len(tgt_tokens), :len(src_tokens)]
                
                # plot cross-attention heatmap
                if args.plot_heatmaps:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    graph = sns.heatmap(torch.log(cross_att), xticklabels=src_tokens, yticklabels=tgt_tokens)
                    graph.set_xticklabels(graph.get_xmajorticklabels(), fontsize=12)
                    graph.set_yticklabels(graph.get_ymajorticklabels(), fontsize=12)
                    
                    plt.savefig(os.path.join(os.path.dirname(args.save), 'heatmap_{}'.format(j * batch_size + i)))
                    plt.show()

                # remove eos token if present
                if tgt_tokens[-1] in numericalizer._tokenizer.all_special_tokens:
                    tgt_tokens = tgt_tokens[:-1]
                
                # TODO _tokenizer should not be private
                if args.replace_qp:
                    text, is_replaced = replace_quoted_params(src_tokens, tgt_tokens, numericalizer._tokenizer, cross_att, model.tgt_lang)
                    if not is_replaced and args.force_replace_qp:
                        text = force_replace_quoted_params(src_tokens, tgt_tokens, numericalizer._tokenizer, cross_att)
                else:
                    text = numericalizer._tokenizer.convert_tokens_to_string(tgt_tokens)
                
                all_tgt_strings.append(text)
            
            with numericalizer._tokenizer.as_target_tokenizer():
                partial_batch_prediction_ids = numericalizer._tokenizer.batch_encode_plus(all_tgt_strings, padding=True, return_tensors='pt').data['input_ids']

            if output_confidence_features or output_confidence_scores:
                partial_batch_confidence_features = model.confidence_features(batch=batch, predictions=partial_batch_prediction_ids, mc_dropout_num=args.mc_dropout_num)
                
            partial_batch_prediction = numericalizer.reverse(partial_batch_prediction_ids, 'answer')
            # post-process predictions
            for i in range(len(partial_batch_prediction)):
                partial_batch_prediction[i] = task.postprocess_prediction(batch_example_ids[(i//args.num_outputs[hyperparameter_idx]) % batch_size], partial_batch_prediction[i])
            # put them into the right array
            for i in range(len(partial_batch_prediction)):
                batch_prediction[(i//args.num_outputs[hyperparameter_idx]) % batch_size].append(partial_batch_prediction[i])
                if output_confidence_features or output_confidence_scores:
                    batch_confidence_features[(i//args.num_outputs[hyperparameter_idx]) % batch_size].append(partial_batch_confidence_features[i])
        
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
        if args.override_confidence_labels:
            for i, example in enumerate(confidence_features):
                for confidence in example:
                    confidence.label = (answers[i] == args.override_confidence_labels)
    if output_confidence_scores:
        output.confidence_scores = []
        for estimator in confidence_estimators:
            confidence_scores = estimator.estimate(confidence_features)
            output.confidence_scores.append(confidence_scores)

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
