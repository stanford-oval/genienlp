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
import ujson

import requests
from tqdm import tqdm

from ..data_utils.numericalizer import TransformerNumericalizer
from ..util import replace_capturing_group

logger = logging.getLogger(__name__)


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
class TransformerSeq2Seq():

    numericalizer: TransformerNumericalizer

    @classmethod
    def load(cls, save_directory: str, args):
        """
        Loads a GenieModel (in Genie format, not HuggingFace's transformers) and its
        accompanying Numericalizer (not HuggingFace's tokenizers) from `save_directory`, which is a path
        """
        model = cls(args, save_directory)
        

        return model

    def __init__(self, args, save_directory):
        """
        If `save_directory` is None, will initialize a new model and numericalizer, otherwise, will load them from `save_directory`
        """
        self.args = args

        self.numericalizer = TransformerNumericalizer(
            self.args.pretrained_model,
            args,
            save_dir=save_directory,
        )

    def validate(
        self,
        data_iterator,
        task,
        eval_dir=None,
        original_order=None,
    ):
        if self.args.e2e_dialogue_evaluation:
            return self.validate_e2e_dialogues(
                data_iterator, task, eval_dir, original_order
            )
        else:
            return self.validate_batch(
                data_iterator,
                task,
                original_order,
            )

    def validate_batch(
        self,
        data_iterator,
        task,
        original_order=None,
    ):
        """
        Inputs:
            original_order: List of indices. If provided, we will sort the results according to this order
        Outputs: (loss, predictions, answers, contexts)
            loss
            predictions: a List of Lists of strings
            answers
            contexts
        """
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
            batch_answer = self.numericalizer.reverse(batch.answer.value.data, 'answer')
            batch_answer = [
                task.postprocess_prediction(batch_example_ids[i], batch_answer[i]) for i in range(len(batch_answer))
            ]
            answers += batch_answer
            batch_context = self.numericalizer.reverse(batch.context.value.data, 'context')
            contexts += batch_context

            for hyperparameter_idx in range(len(self.args.temperature)):
                print(batch)
                raise NotImplementedError() # TODO
                partial_batch_prediction = self.generate(batch)

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



        output = ValidationOutput()
        output.example_ids, output.predictions, output.answers, output.contexts = (
            example_ids,
            predictions,
            answers,
            contexts,
        )
        return output

    def validate_e2e_dialogues(
        self, data_iterator, task, eval_dir=None, original_order=None
    ):
        """
        Inputs:
            original_order: List of indices. If provided, we will sort the results according to this order
        Outputs: (loss, predictions, answers, contexts)
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

            response = requests.get("http://127.0.0.1:7878/generate", json={"language":self.args.language, "task_input":input_text, "model":"gpt-4"})
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

        output = ValidationOutput()

        output.example_ids, output.predictions, output.answers, output.contexts = (
            example_ids,
            predictions,
            answers,
            contexts,
        )

        return output

    def interact_e2e_dialogues(self, task, eval_dir=None, original_order=None):
        """
        Inputs:
            original_order: List of indices. If provided, we will sort the results according to this order
        Outputs: (loss, predictions, answers, contexts)
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

                partial_batch_prediction = self.generate(input_text)

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
                            dialogue_state, knowledge, api_names, self.language, dial_id, turn_id
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

        output = ValidationOutput()
        output.example_ids, output.predictions, output.answers, output.contexts = (
            example_ids,
            predictions,
            answers,
            contexts,
        )

        return output
    
    def generate(self, input_text:str):
        response = requests.get("http://127.0.0.1:7878/generate", json={"language":self.args.language, "task_input":input_text, "model":"gpt-4"})
        return response.json()["task_output"]