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
import copy
import logging
import re
import sys
import time
from collections import defaultdict

import torch
import ujson
from BiToD.evaluate import r_en_API_MAP, span2dict, state2api
from BiToD.knowledgebase import api
from BiToD.preprocess import API_MAP, knowledge2span, read_require_slots, state2span

from .data_utils.example import NumericalizedExamples, SequentialField
from .data_utils.progbar import progress_bar
from .metrics import calculate_and_reduce_metrics
from .models import TransformerForSequenceClassification, TransformerForTokenClassification
from .util import GenerationOutput, merge_translated_sentences

logger = logging.getLogger(__name__)


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
    interactive=False,
):
    if args.bitod_e2e_evaluation:
        if interactive:
            return generate_with_seq2seq_model_for_dialogue_interactive(
                model,
                numericalizer,
                task,
                args,
                output_predictions_only=output_predictions_only,
                original_order=original_order,
                disable_progbar=disable_progbar,
            )
        else:
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

    elif isinstance(model, (TransformerForTokenClassification, TransformerForSequenceClassification)):
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


def replace_match(input, re_pattern, replacement):
    match = re_pattern.search(input).group(1).strip()
    return input.replace(match, replacement)


def generate_with_seq2seq_model_for_dialogue_interactive(
    model,
    numericalizer,
    task,
    args,
    output_predictions_only=False,
    original_order=None,
    disable_progbar=True,
) -> GenerationOutput:

    bitod_preds = dict()

    predictions = []
    example_ids = []
    answers = []
    contexts = []

    device = model.device

    required_slots = read_require_slots()
    required_slots = {API_MAP[k]: v for k, v in required_slots.items()}
    api_names = list(required_slots.keys())

    dial_id = 'none'
    turn_id = 1
    dialogue_state = {}
    new_state_text = 'null'
    new_knowledge_text = 'null'
    active_api = None
    bitod_preds[dial_id] = {"turns": defaultdict(dict), "API": defaultdict(dict)}

    convo_history = []
    convo_window = 3

    train_target = 'response'

    next_target = {'dst': 'api', 'api': 'response', 'response': 'dst'}

    while True:

        try:

            batch_prediction = []

            # becomes dst for first turn
            train_target = next_target[train_target]

            hyperparameter_idx = 0

            if train_target == 'dst':

                print('convo_history: ', convo_history)
                # Hello, I am looking for a restaurant with Vegan Options.
                # construct new input
                raw_user_input = input('USER: ')
                if raw_user_input == 'RESET':
                    generate_with_seq2seq_model_for_dialogue_interactive(
                        model,
                        numericalizer,
                        task,
                        args,
                        output_predictions_only=False,
                        original_order=None,
                        disable_progbar=True,
                    )
                    break
                elif raw_user_input == 'END':
                    break

                raw_user_input = 'USER: ' + raw_user_input.strip()

                convo_history.append(raw_user_input)

                input_text = f'DST: <state> {new_state_text} <history> {" ".join(convo_history[-convo_window:])}'

            elif train_target == 'api':
                new_state_text = state2span(dialogue_state, required_slots)

                # replace state
                input_text = f'API: <state> {new_state_text} <history> {" ".join(convo_history[-convo_window:])}'

            elif train_target == 'response':

                input_text = f'Response: <knowledge> {new_knowledge_text} <state> {new_state_text} <history> {" ".join(convo_history[-convo_window:])}'

            else:
                raise ValueError(f'Invalid train_target: {train_target}')

            tokenized_contexts = numericalizer.encode_batch([input_text], field_name='context', features=None)[0]

            numericalized_turn = NumericalizedExamples(
                example_id=[str(turn_id)],
                context=SequentialField(
                    value=torch.tensor([tokenized_contexts.value], device=device),
                    length=torch.tensor([tokenized_contexts.length], device=device),
                    limited=torch.tensor([tokenized_contexts.limited], device=device),
                    feature=None,
                ),
                answer=SequentialField(
                    value=None,
                    length=None,
                    limited=None,
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
            partial_batch_prediction = task.postprocess_prediction(turn_id, partial_batch_prediction)

            # put them into the right array
            batch_prediction.append([partial_batch_prediction])

            predictions += batch_prediction

            if train_target == 'dst':
                # update dialogue_state
                lev = predictions[-1][0].strip()
                state_update = span2dict(lev, api_names)
                for api_name in state_update:
                    active_api = api_name
                    if api_name not in dialogue_state:
                        dialogue_state[api_name] = state_update[api_name]
                    else:
                        dialogue_state[api_name].update(state_update[api_name])

                #### save latest state
                state_to_record = copy.deepcopy(dialogue_state)
                state_to_record = {r_en_API_MAP.get(k, k): v for k, v in state_to_record.items()}
                bitod_preds[dial_id]["turns"][str(turn_id)]["state"] = state_to_record
                ####

            elif train_target == 'api':
                new_knowledge_text = 'null'
                constraints = {}

                api_name = active_api if active_api else 'None'

                do_api_call = predictions[-1][0].strip()

                if do_api_call == 'yes':
                    # make api call if required
                    api_name = active_api
                    # do api call
                    if api_name in dialogue_state:
                        constraints = state2api(dialogue_state[api_name])

                        try:
                            msg = api.call_api(
                                r_en_API_MAP.get(api_name, api_name),
                                constraints=[constraints],
                            )
                        except Exception as e:
                            print(f'Error: {e}')
                            print(f'Failed API call with api_name: {api_name} and constraints: {constraints}')
                            msg = [0, 0]

                        domain = api_name.split(" ")[0]

                        knowledge = defaultdict(dict)
                        if int(msg[1]) <= 0:
                            new_knowledge_text = f'( {domain} ) Message = No item available.'
                        else:
                            # why does it only choose the first; does the same happen for training data?
                            knowledge[domain].update(msg[0])
                            new_knowledge_text = knowledge2span(knowledge)

                elif do_api_call == 'no':
                    # do nothing
                    pass
                else:
                    logger.error(
                        f'API call should be either yes or no but got {do_api_call}; seems model is still training so we assume it\'s a no'
                    )

                #### save latest api results and constraints
                bitod_preds[dial_id]["turns"][str(turn_id)]["api"] = new_knowledge_text
                bitod_preds[dial_id]["API"][r_en_API_MAP.get(api_name, api_name)] = copy.deepcopy(constraints)
                ####

            if train_target == 'response':
                #### save latest response
                bitod_preds[dial_id]["turns"][str(turn_id)]["response"] = predictions[-1]
                ####

                convo_history.append(predictions[-1][0])

        except KeyboardInterrupt:
            break

    with open(f"{int(time.time())}_bitod_preds.json", 'w') as fout:
        ujson.dump(bitod_preds, fout, indent=2, ensure_ascii=False)

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

    # history_re = re.compile('<history> (.*?)(?:$|<)')
    # last_system_re = re.compile('SYSTEM: (.*?)(?:USER:|$)')

    state_re = re.compile('<state> (.*?)(?:$|<)')
    knowledge_re = re.compile('<knowledge> (.*?)(?:$|<)')

    bitod_preds = dict()

    predictions = []
    example_ids = []
    answers = []
    contexts = []

    cur_dial_id = ''

    device = model.device

    required_slots = read_require_slots()
    required_slots = {API_MAP[k]: v for k, v in required_slots.items()}
    api_names = list(required_slots.keys())

    for k, turn in enumerate(progress_bar(data_iterator, desc='Generating', disable=disable_progbar)):
        batch_size = len(turn.example_id)
        assert batch_size == 1
        batch_prediction = []
        batch_example_ids = turn.example_id

        example_ids += batch_example_ids

        task_name, dial_id, turn_id, train_target = example_ids[-1].split('/')
        turn_id = int(turn_id)

        if cur_dial_id != dial_id:
            # new dialogue
            cur_dial_id = dial_id
            dialogue_state = {}
            new_state_text = 'null'
            new_knowledge_text = 'null'
            active_api = None
            bitod_preds[dial_id] = {"turns": defaultdict(dict), "API": defaultdict(dict)}

        special_tokens = numericalizer._tokenizer.all_special_tokens
        batch_tokens = numericalizer.convert_ids_to_tokens(turn.context.value.data, skip_special_tokens=False)
        batch_context = []
        # remove only beginning and trailing special tokens
        # otherwise the sep_token added between context and question will be lost
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

        if not output_predictions_only:
            batch_answer = numericalizer.reverse(turn.answer.value.data, 'answer')
            batch_answer = [
                task.postprocess_prediction(batch_example_ids[i], batch_answer[i]) for i in range(len(batch_answer))
            ]
            answers += batch_answer

        # iterate through turns
        hyperparameter_idx = 0

        # we always use gold history following common practice

        # if first_turn:
        #     # first turn is always dst
        #     assert train_target == 'dst'
        #     numericalized_turn = NumericalizedExamples(
        #         example_id=[turn.example_id[0]],
        #         context=SequentialField(
        #             value=turn.context.value[[0]],
        #             length=turn.context.length[[0]],
        #             limited=turn.context.limited[[0]],
        #             feature=None,
        #         ),
        #         answer=SequentialField(
        #             value=turn.answer.value[[0]],
        #             length=turn.answer.value[[0]],
        #             limited=turn.answer.value[[0]],
        #             feature=None,
        #         ),
        #     )
        # else:
        if train_target == 'dst':
            input_text = replace_match(contexts[-1], state_re, new_state_text)

            ## if you want to use predicted response instead of gold uncomment the following
            # last_sys_pred = predictions[-1][0].strip()
            # input_text = replace_match(input_text, last_system_re, last_sys_pred)

        elif train_target == 'api':
            new_state_text = state2span(dialogue_state, required_slots)

            # replace state
            input_text = replace_match(contexts[-1], state_re, new_state_text)

        elif train_target == 'response':
            # replace state
            input_text = replace_match(contexts[-1], state_re, new_state_text)

            # replace knowledge
            input_text = replace_match(input_text, knowledge_re, new_knowledge_text)

        else:
            raise ValueError(f'Invalid train_target: {train_target}')

        tokenized_contexts = numericalizer.encode_batch([input_text], field_name='context', features=None)[0]

        numericalized_turn = NumericalizedExamples(
            example_id=[turn.example_id[0]],
            context=SequentialField(
                value=torch.tensor([tokenized_contexts.value], device=device),
                length=torch.tensor([tokenized_contexts.length], device=device),
                limited=torch.tensor([tokenized_contexts.limited], device=device),
                feature=None,
            ),
            answer=SequentialField(
                value=turn.answer.value[[0]],
                length=turn.answer.value[[0]],
                limited=turn.answer.value[[0]],
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
        partial_batch_prediction = task.postprocess_prediction(batch_example_ids[0], partial_batch_prediction)

        # put them into the right array
        batch_prediction.append([partial_batch_prediction])

        predictions += batch_prediction

        if train_target == 'dst':
            # update dialogue_state
            lev = predictions[-1][0].strip()
            state_update = span2dict(lev, api_names)
            for api_name in state_update:
                active_api = api_name
                if api_name not in dialogue_state:
                    dialogue_state[api_name] = state_update[api_name]
                else:
                    dialogue_state[api_name].update(state_update[api_name])

            #### save latest state
            state_to_record = copy.deepcopy(dialogue_state)
            state_to_record = {r_en_API_MAP.get(k, k): v for k, v in state_to_record.items()}
            bitod_preds[dial_id]["turns"][str(turn_id)]["state"] = state_to_record
            ####

        elif train_target == 'api':
            new_knowledge_text = 'null'
            constraints = {}

            api_name = active_api if active_api else 'None'

            do_api_call = predictions[-1][0].strip()

            if do_api_call == 'yes':
                # make api call if required
                api_name = active_api
                # do api call
                if api_name in dialogue_state:
                    constraints = state2api(dialogue_state[api_name])

                    try:
                        msg = api.call_api(
                            r_en_API_MAP.get(api_name, api_name),
                            constraints=[constraints],
                        )
                    except Exception as e:
                        print(f'Error: {e}')
                        print(f'Failed API call with api_name: {api_name} and constraints: {constraints}')
                        msg = [0, 0]

                    domain = api_name.split(" ")[0]

                    knowledge = defaultdict(dict)
                    if int(msg[1]) <= 0:
                        new_knowledge_text = f'( {domain} ) Message = No item available.'
                    else:
                        # why does it only choose the first; does the same happen for training data?
                        knowledge[domain].update(msg[0])
                        new_knowledge_text = knowledge2span(knowledge)

            elif do_api_call == 'no':
                # do nothing
                pass
            else:
                logger.error(
                    f'API call should be either yes or no but got {do_api_call}; seems model is still training so we assume it\'s a no'
                )

            #### save latest api results and constraints
            bitod_preds[dial_id]["turns"][str(turn_id)]["api"] = new_knowledge_text
            bitod_preds[dial_id]["API"][r_en_API_MAP.get(api_name, api_name)] = copy.deepcopy(constraints)
            ####

        if train_target == 'response':
            #### save latest response
            bitod_preds[dial_id]["turns"][str(turn_id)]["response"] = predictions[-1]
            ####

    with open(f"{int(time.time())}_bitod_preds.json", 'w') as fout:
        ujson.dump(bitod_preds, fout, indent=2, ensure_ascii=False)

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

    if total_loss is not None:
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


def print_results(results, num_print):
    print()

    values = list(results.values())
    num_examples = len(values[0])

    # examples are sorted by length
    # to get good diversity, get half of examples from second quartile
    start = int(num_examples / 4)
    end = start + int(num_print / 2)
    first_list = [val[start:end] for val in values]

    # and the other half from fourth quartile
    start = int(3 * num_examples / 4)
    end = start + num_print - int(num_print / 2)
    second_list = [val[start:end] for val in values]

    # join examples
    processed_values = [first + second for first, second in zip(first_list, second_list)]

    for ex_idx in range(len(processed_values[0])):
        for key_idx, key in enumerate(results.keys()):
            value = processed_values[key_idx][ex_idx]
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

        output = generate_with_model(model, val_iter, numericalizer, task, args)

        # loss is already calculated
        metrics_to_compute = [metric for metric in task.metrics if metric not in ['loss']]
        metrics = calculate_and_reduce_metrics(
            output.predictions, output.answers, metrics_to_compute, args.reduce_metrics, model.tgt_lang
        )

        results = {'beam search': output.predictions, 'answer': output.answers, 'context': output.contexts}

        print_results(results, num_print)

        return output, metrics
