#
# Copyright (c) 2020 The Board of Trustees of the Leland Stanford Junior University
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
import os
from collections import defaultdict
from typing import List, Optional

import dialogues
import torch
import ujson
from dateparser.languages import default_loader
import requests
from transformers import MarianTokenizer, PreTrainedModel
from tqdm import tqdm

from ..data_utils.example import NumericalizedExamples, SequentialField
from ..data_utils.numericalizer import TransformerNumericalizer
from ..util import replace_capturing_group

logger = logging.getLogger(__name__)


class GenieModel(PreTrainedModel):
    numericalizer: TransformerNumericalizer

    @classmethod
    def load(cls, save_directory: str, *model_args, **kwargs):
        """
        Loads a GenieModel (in Genie format, not HuggingFace's transformers) and its
        accompanying Numericalizer (not HuggingFace's tokenizers) from `save_directory`, which is a path
        """
        # TODO remove kwargs and take individual inputs instead
        args = kwargs.pop("args", None)
        tasks = kwargs.pop("tasks", None)
        vocab_sets = kwargs.pop("vocab_sets", None)

        model = cls(args=args, tasks=tasks, vocab_sets=vocab_sets, save_directory=save_directory, *model_args, **kwargs)
        

        return model


class ValidationOutput(object):
    """
    Contains all the information that model's validate() method may output
    """

    def __init__(
        self,
        loss: Optional[float] = None,
        example_ids: Optional[List] = None,
        predictions: Optional[List] = None,
        raw_predictions: Optional[List] = None,
        answers: Optional[List] = None,
        contexts: Optional[List] = None,
    ):
        self.loss = loss
        self.example_ids = example_ids
        self.predictions = predictions
        self.raw_predictions = raw_predictions
        self.answers = answers
        self.contexts = contexts


# TransformerSeq2Seq inherits from this model
class GenieModelForGeneration(GenieModel):
    def numericalize_example(self, input_text, turn_id):
        if isinstance(input_text, str):
            input_text = [input_text]
        tokenized_contexts = self.numericalizer.encode_batch(input_text, field_name='context', features=None)[0]

        numericalized_turn = NumericalizedExamples(
            example_id=[str(turn_id)],
            context=SequentialField(
                value=torch.tensor([tokenized_contexts.value]),
                length=torch.tensor([tokenized_contexts.length]),
                limited=torch.tensor([tokenized_contexts.limited]),
                feature=None,
            ),
            answer=SequentialField(value=None, length=None, limited=None, feature=None),
        )

        return numericalized_turn

    def validate(
        self,
        data_iterator,
        task,
        eval_dir=None,
        output_predictions_only=False,
        original_order=None,
        disable_progbar=True,
        **kwargs,
    ):
        if self.args.e2e_dialogue_evaluation:
            return self.validate_e2e_dialogues(
                data_iterator, task, eval_dir, output_predictions_only, original_order
            )
        else:
            return self.validate_batch(
                data_iterator,
                task,
                output_predictions_only,
                original_order,
                disable_progbar,
            )

    def validate_batch(
        self,
        data_iterator,
        task,
        output_predictions_only=False,
        original_order=None,
        disable_progbar=True,
    ):
        """
        Inputs:
            original_order: List of indices. If provided, we will sort the results according to this order
        Outputs: predictions if `output_predictions_only` == True, (loss, predictions, answers, contexts) otherwise
            loss
            predictions: a List of Lists of strings
            answers
            contexts
        """
        total_loss = 0.0 if 'loss' in task.metrics else None
        predictions = []
        raw_predictions = []
        example_ids = []
        answers = []
        contexts = []

        for batch in tqdm(data_iterator, desc='Generating'):
            batch_size = len(batch.example_id)
            batch_prediction = [[] for _ in range(batch_size)]
            batch_raw_prediction = [[] for _ in range(batch_size)]
            batch_example_ids = batch.example_id

            example_ids += batch_example_ids
            if not output_predictions_only:
                batch_answer = self.numericalizer.reverse(batch.answer.value.data, 'answer')
                batch_answer = [
                    task.postprocess_prediction(batch_example_ids[i], batch_answer[i]) for i in range(len(batch_answer))
                ]
                answers += batch_answer
                batch_context = self.numericalizer.reverse(batch.context.value.data, 'context')
                contexts += batch_context

            if total_loss is not None:
                loss = self.forward(batch, train=True).loss.item()
                total_loss += loss

            for hyperparameter_idx in range(len(self.args.temperature)):
                generated = self.generate(
                    batch,
                    max_output_length=self.args.max_output_length,
                    min_output_length=self.args.min_output_length,
                    num_outputs=self.args.num_outputs[hyperparameter_idx],
                    temperature=self.args.temperature[hyperparameter_idx]
                    if self.args.temperature[hyperparameter_idx] > 0
                    else 1.0,
                    repetition_penalty=self.args.repetition_penalty[hyperparameter_idx],
                    top_k=self.args.top_k[hyperparameter_idx],
                    top_p=self.args.top_p[hyperparameter_idx],
                    num_beams=self.args.num_beams[hyperparameter_idx],
                    num_beam_groups=self.args.num_beam_groups[hyperparameter_idx],
                    diversity_penalty=self.args.diversity_penalty[hyperparameter_idx],
                    no_repeat_ngram_size=self.args.no_repeat_ngram_size[hyperparameter_idx],
                    do_sample=self.args.temperature[hyperparameter_idx] != 0,  # if temperature==0, we do not sample
                )
                partial_batch_prediction_ids = generated.sequences
                partial_batch_words = None

                if getattr(task, 'need_attention_scores', False):
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
                    kwargs = {
                        'numericalizer': self.numericalizer,
                        'cross_attentions': cross_attentions,
                    }

                    partial_batch_prediction_ids, partial_batch_words = task.batch_postprocess_prediction_ids(
                        batch_example_ids, batch.context.value.data, partial_batch_prediction_ids, **kwargs
                    )

                # MarianTokenizer uses two different spm models for encoding source and target languages.
                # in almond_translate we postprocess text with alignment which produces code-switched sentences.
                # encoding a code-switched sentence with either spm will omit tokens from the other language
                # so we have to return both the processed and encoded text.
                # we need to return encoded text too since confidence_features requires ids
                if isinstance(self.numericalizer._tokenizer, MarianTokenizer) and partial_batch_words:
                    partial_batch_prediction = partial_batch_words
                else:
                    partial_batch_prediction = self.numericalizer.reverse(partial_batch_prediction_ids, 'answer')

                def get_example_index(i):
                    return (i // self.args.num_outputs[hyperparameter_idx]) % batch_size

                # post-process predictions
                for i in range(len(partial_batch_prediction)):
                    partial_batch_prediction[i] = task.postprocess_prediction(
                        batch_example_ids[get_example_index(i)], partial_batch_prediction[i]
                    )

                # put them into the right array
                for i in range(len(partial_batch_prediction)):
                    batch_prediction[get_example_index(i)].append(partial_batch_prediction[i])

            predictions += batch_prediction
            raw_predictions += batch_raw_prediction

        if total_loss is not None:
            total_loss /= len(example_ids)

        if original_order is not None:
            # sort back to the original order
            original_order, example_ids, predictions, raw_predictions, answers, contexts = [
                list(a)
                for a in tuple(
                    zip(
                        *sorted(
                            list(
                                zip(
                                    original_order,
                                    example_ids,
                                    predictions,
                                    raw_predictions,
                                    answers,
                                    contexts,
                                )
                            )
                        )
                    )
                )
            ]



        output = ValidationOutput(loss=total_loss)

        if output_predictions_only:
            output.predictions = predictions
        else:
            output.example_ids, output.predictions, output.answers, output.contexts = (
                example_ids,
                predictions,
                answers,
                contexts,
            )
        return output

    def validate_e2e_dialogues(
        self, data_iterator, task, eval_dir=None, output_predictions_only=False, original_order=None
    ):
        """
        Inputs:
            original_order: List of indices. If provided, we will sort the results according to this order
        Outputs: predictions if `output_predictions_only` == True, (loss, predictions, answers, contexts) otherwise
            loss
            predictions: a List of Lists of strings
            answers
            contexts
        """
        dataset_class = getattr(dialogues, task.dataset_name)
        dataset = dataset_class()

        special_tokens = self.numericalizer._tokenizer.all_special_tokens

        e2e_dialogue_preds = dict()
        predictions = []
        example_ids = []
        answers = []
        contexts = []

        cur_dial_id = ''

        for turn in tqdm(data_iterator, desc='Generating'):
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
                knowledge = defaultdict(dict)
                new_knowledge_text = 'null'
                new_actions_text = 'null'
                api_names = []
                state_update = {}
                e2e_dialogue_preds[dial_id] = {"turns": defaultdict(dict), "API": defaultdict(dict)}

            batch_context = []
            batch_tokens = self.numericalizer.convert_ids_to_tokens(turn.context.value.data, skip_special_tokens=False)

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

                batch_context.append(self.numericalizer._tokenizer.convert_tokens_to_string(text).replace(",", "，"))

            contexts += batch_context

            if not output_predictions_only:
                batch_answer = self.numericalizer.reverse(turn.answer.value.data, 'answer')
                batch_answer = [
                    task.postprocess_prediction(batch_example_ids[i], batch_answer[i]) for i in range(len(batch_answer))
                ]
                answers += batch_answer

            new_state_text = dataset.state2span(dialogue_state)

            if train_target == 'dst':
                input_text = replace_capturing_group(contexts[-1], dataset.state_re, new_state_text)

                ## we always use gold history following common practice
                ## if you want to use predicted response instead of gold uncomment the following
                # last_sys_pred = predictions[-1][0].strip()
                # input_text = replace_match(input_text, last_system_re, last_sys_pred)

            elif train_target == 'api':
                # replace state
                input_text = replace_capturing_group(contexts[-1], dataset.state_re, new_state_text)

            elif train_target == 'da':
                # replace state
                input_text = replace_capturing_group(contexts[-1], dataset.state_re, new_state_text)

                # replace knowledge
                input_text = replace_capturing_group(input_text, dataset.knowledge_re, new_knowledge_text)

            elif train_target == 'rg':

                # replace actions
                input_text = replace_capturing_group(contexts[-1], dataset.actions_re, new_actions_text)

            else:
                raise ValueError(f'Invalid train_target: {train_target}')

            # replace old context with updated
            contexts[-1] = input_text

            response = requests.get("http://127.0.0.1:7878/generate", json={"language":"en", "task_input":input_text, "model":"gpt-4"})
            partial_batch_prediction = response.json()["task_output"]
            print(partial_batch_prediction)

            if train_target == 'da':
                partial_batch_prediction = dataset.postprocess_prediction(
                    partial_batch_prediction, knowledge=knowledge, lang=self.args.language
                )

            partial_batch_prediction = task.postprocess_prediction(batch_example_ids[0], partial_batch_prediction)

            # put them into the right array
            batch_prediction.append([partial_batch_prediction])

            predictions += batch_prediction

            if train_target == 'dst':
                # update dialogue_state
                lev = predictions[-1][0].strip()
                state_update = dataset.span2state(lev)
                dataset.update_state(state_update, dialogue_state)

                #### save latest state
                e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["state"] = dataset.state2span(dialogue_state)
                ####

            elif train_target == 'api':
                do_api_call = predictions[-1][0].strip()

                if do_api_call == 'yes':
                    # make api call
                    if state_update:
                        api_names = list(state_update.keys())
                    new_knowledge_text, constraints = dataset.make_api_call(
                        dialogue_state, knowledge, api_names, self.args.language, dial_id, turn_id
                    )
                    #### save latest api constraints
                    e2e_dialogue_preds[dial_id]["API"] = copy.deepcopy(constraints)
                    ####

                elif do_api_call == 'no':
                    # do nothing
                    pass
                else:
                    logger.error(
                        f'API call should be either yes or no but got {do_api_call}. Seems model has not learnt the task yet. For now we assume it\'s a no'
                    )

                #### save latest api results
                e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["api"] = new_knowledge_text
                ####

            elif train_target == 'da':
                new_actions_text = predictions[-1][0]
                #### save latest actions
                e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["actions"] = predictions[-1][0]
                ####

            elif train_target == 'rg':
                #### save latest response
                e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["response"] = predictions[-1]
                ####

        if eval_dir:
            with open(os.path.join(eval_dir, 'e2e_dialogue_preds.json'), 'w') as fout:
                ujson.dump(e2e_dialogue_preds, fout, indent=2, ensure_ascii=False)

        if original_order is not None:
            # sort back to the original order
            original_order, example_ids, predictions, answers, contexts = [
                list(a) for a in tuple(zip(*sorted(list(zip(original_order, example_ids, predictions, answers, contexts)))))
            ]

        # TODO calculate and return loss
        loss = None
        output = ValidationOutput(loss=loss)

        if output_predictions_only:
            output.predictions = predictions
        else:
            output.example_ids, output.predictions, output.answers, output.contexts = (
                example_ids,
                predictions,
                answers,
                contexts,
            )

        return output

    def interact_e2e_dialogues(self, task, eval_dir=None, output_predictions_only=False, original_order=None):
        """
        Inputs:
            original_order: List of indices. If provided, we will sort the results according to this order
        Outputs: predictions if `output_predictions_only` == True, (loss, predictions, answers, contexts) otherwise
            loss
            predictions: a List of Lists of strings
            answers
            contexts
        """
        # lazily import termcolor
        from termcolor import colored

        dataset_class = getattr(dialogues, task.dataset_name)
        dataset = dataset_class()

        e2e_dialogue_preds = dict()

        predictions = []
        example_ids = []
        answers = []
        contexts = []

        hyperparameter_idx = 0

        args = self.args

        NEXT_TARGET = {'dst': 'api', 'api': 'da', 'da': 'rg', 'rg': 'dst'}

        INIT_SYS_MESSAGE = {
            'en': 'Hello! How can I help you today?',
            'fa': 'سلام! امروز چطور می توانم به شما کمک کنم؟',
            'zh': '你好！ 我今天能帮到你什么？',
        }

        # new dialogue
        train_target = 'dst'
        dial_id = '0'
        turn_id = 0
        dialogue_state = {}
        knowledge = defaultdict(dict)
        new_state_text = 'null'
        new_knowledge_text = 'null'
        new_actions_text = 'null'
        api_names = []
        user_history = []
        system_history = []
        state_update = {}
        e2e_dialogue_preds[dial_id] = {"turns": defaultdict(dict), "API": defaultdict(dict)}

        while True:
            try:
                batch_prediction = []

                if train_target == 'dst':
                    if system_history:
                        print(colored('SYSTEM: ' + f'{predictions[-1][0]}', 'red', attrs=['bold']))
                    else:
                        print(colored('SYSTEM: ' + f'{INIT_SYS_MESSAGE[self.args.language]}', 'red', attrs=['bold']))

                    # construct new input
                    raw_user_input = input(colored('USER: ', 'green', attrs=['bold'])).strip()
                    if raw_user_input == 'RESET':
                        # next dialogue
                        train_target = 'dst'
                        dial_id = str(int(dial_id) + 1)
                        turn_id = 0
                        dialogue_state = {}
                        knowledge = defaultdict(dict)
                        new_state_text = 'null'
                        new_knowledge_text = 'null'
                        new_actions_text = 'null'
                        api_names = []
                        user_history = []
                        system_history = []
                        state_update = {}
                        e2e_dialogue_preds[dial_id] = {"turns": defaultdict(dict), "API": defaultdict(dict)}
                        continue
                    elif raw_user_input == 'EXIT':
                        break
                    elif raw_user_input == 'STATE':
                        print(f'dialogue state: {dialogue_state}')
                        continue
                    elif raw_user_input == 'KNOWLEDGE':
                        print(f'knowledge: {knowledge}')
                        continue
                    elif raw_user_input == 'USER_HISTORY':
                        print(f'user history: {user_history}')
                        continue
                    elif raw_user_input == 'SYSTEM_HISTORY':
                        print(f'system history: {system_history}')
                        continue
                    elif raw_user_input == 'ACTIONS':
                        print(f'agent actions: {new_actions_text}')
                        continue

                    user_history.append(dataset.user_token + ' ' + raw_user_input)

                    ## record user input
                    e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["user_input"] = raw_user_input

                    turn_id += 1

                input_text = dataset.construct_input(
                    train_target,
                    state=new_state_text,
                    user_history=user_history,
                    system_history=system_history,
                    knowledge=new_knowledge_text,
                    actions=new_actions_text,
                )

                ## record model input
                e2e_dialogue_preds[dial_id]["turns"][str(turn_id)][f"model_input_{train_target}"] = input_text

                numericalized_turn = self.numericalize_example(input_text, turn_id, device)
                generated = self.generate(
                    numericalized_turn,
                    max_output_length=args.max_output_length,
                    min_output_length=args.min_output_length,
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
                partial_batch_prediction = self.numericalizer.reverse(partial_batch_prediction_ids, 'answer')[0]

                if train_target == 'da':
                    partial_batch_prediction = dataset.postprocess_prediction(
                        partial_batch_prediction, knowledge=knowledge, lang=self.args.language
                    )

                partial_batch_prediction = task.postprocess_prediction(turn_id, partial_batch_prediction)

                # put them into the right array
                batch_prediction.append([partial_batch_prediction])

                # record model output
                e2e_dialogue_preds[dial_id]["turns"][str(turn_id)][f"model_output_{train_target}"] = partial_batch_prediction

                predictions += batch_prediction

                if train_target == 'dst':
                    # update dialogue_state
                    lev = predictions[-1][0].strip()
                    state_update = dataset.span2state(lev)
                    dataset.update_state(state_update, dialogue_state)
                    new_state_text = dataset.state2span(dialogue_state)

                    #### save latest state
                    e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["state"] = new_state_text
                    ####

                elif train_target == 'api':
                    do_api_call = predictions[-1][0].strip()

                    if do_api_call == 'yes':
                        # make api call
                        if state_update:
                            api_names = list(state_update.keys())
                        new_knowledge_text, constraints = dataset.make_api_call(
                            dialogue_state, knowledge, api_names, src_lang, dial_id, turn_id
                        )
                        #### save latest api constraints
                        e2e_dialogue_preds[dial_id]["API"] = copy.deepcopy(constraints)
                        ####

                    elif do_api_call == 'no':
                        # do nothing
                        pass
                    else:
                        logger.error(
                            f'API call should be either yes or no but got {do_api_call}. Seems model has not learnt the task yet. For now we assume it\'s a no'
                        )

                    #### save latest api results
                    e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["api"] = new_knowledge_text
                    ####

                elif train_target == 'da':
                    new_actions_text = predictions[-1][0]
                    system_history.append(dataset.system_token + ' ' + new_actions_text)
                    #### save latest actions
                    e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["actions"] = predictions[-1][0]
                    ####

                elif train_target == 'rg':
                    #### save latest response
                    e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["response"] = predictions[-1]
                    ####

                train_target = NEXT_TARGET[train_target]

            except KeyboardInterrupt:
                break

        if eval_dir:
            with open(os.path.join(eval_dir, 'interact_e2e_dialogue_preds.json'), 'w') as fout:
                ujson.dump(e2e_dialogue_preds, fout, indent=2, ensure_ascii=False)

        if original_order is not None:
            # sort back to the original order
            original_order, example_ids, predictions, answers, contexts = [
                list(a) for a in tuple(zip(*sorted(list(zip(original_order, example_ids, predictions, answers, contexts)))))
            ]

        # TODO calculate and return loss
        loss = None
        output = ValidationOutput(loss=loss)

        if output_predictions_only:
            output.predictions = predictions
        else:
            output.example_ids, output.predictions, output.answers, output.contexts = (
                example_ids,
                predictions,
                answers,
                contexts,
            )

        return output
