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
from transformers import AutoConfig, BartForConditionalGeneration, MarianTokenizer, PreTrainedModel

from ..data_utils.example import NumericalizedExamples, SequentialField
from ..data_utils.numericalizer import TransformerNumericalizer
from ..data_utils.progbar import progress_bar
from ..util import adjust_language_code, merge_translated_sentences, replace_capturing_group

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
        model_checkpoint_file = kwargs.pop("model_checkpoint_file", None)
        args = kwargs.pop("args", None)
        device = kwargs.pop("device", None)
        tasks = kwargs.pop("tasks", None)
        vocab_sets = kwargs.pop("vocab_sets", None)

        full_checkpoint_path = os.path.join(save_directory, model_checkpoint_file)
        logger.info(f'Loading the model from {full_checkpoint_path}')
        model = cls(args=args, tasks=tasks, vocab_sets=vocab_sets, save_directory=save_directory, *model_args, **kwargs)
        save_dict = torch.load(full_checkpoint_path, map_location=device)

        # HACK
        # `transformers` version 4.1 changed the name of language modeling head of BartForConditionalGeneration
        # (and therefore its subclass MBartForConditionalGeneration) to lm_head to make it similar to other models
        # like T5. The following will make this change so that genienlp models trained with `transformers`==4.0 can be properly loaded

        # TODO: remove this once we make sure the new paraphraser runs fine
        if (
            'model.lm_head.weight' not in save_dict['model_state_dict']
            and 'model.model.shared.weight' in save_dict['model_state_dict']
            and isinstance(model.model, BartForConditionalGeneration)
        ):
            save_dict['model_state_dict']['model.lm_head.weight'] = save_dict['model_state_dict']['model.model.shared.weight']

        model.load_state_dict(save_dict['model_state_dict'], strict=True)

        return model, save_dict.get('best_decascore')

    def add_new_vocab_from_data(self, tasks, resize_decoder=False):
        old_num_tokens = self.numericalizer.num_tokens
        self.numericalizer.grow_vocab(tasks)
        if self.numericalizer.num_tokens > old_num_tokens:
            logger.info(f'Vocabulary has expanded to {self.numericalizer.num_tokens} tokens')

    def update_language_dependent_configs(self, tgt_lang):
        # we override this method for TransformerSeq2Seq models; otherwise it's a no-op
        pass

    def set_generation_output_options(self, tasks):
        self._output_attentions = any(getattr(task, 'need_attention_scores', False) for task in tasks)
        self._output_scores = False
        self._output_hidden_states = False


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
        confidence_features: Optional[List] = None,
        confidence_scores: Optional[List] = None,
    ):
        self.loss = loss
        self.example_ids = example_ids
        self.predictions = predictions
        self.raw_predictions = raw_predictions
        self.answers = answers
        self.contexts = contexts
        self.confidence_features = confidence_features
        self.confidence_scores = confidence_scores


# TransformerSeq2Seq and TransformerLSTM will inherit from this model
class GenieModelForGeneration(GenieModel):
    def numericalize_example(self, input_text, turn_id, device):
        if isinstance(input_text, str):
            input_text = [input_text]
        tokenized_contexts = self.numericalizer.encode_batch(input_text, field_name='context', features=None)[0]

        numericalized_turn = NumericalizedExamples(
            example_id=[str(turn_id)],
            context=SequentialField(
                value=torch.tensor([tokenized_contexts.value], device=device),
                length=torch.tensor([tokenized_contexts.length], device=device),
                limited=torch.tensor([tokenized_contexts.limited], device=device),
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
        output_confidence_features=False,
        original_order=None,
        confidence_estimators=None,
        disable_progbar=True,
        **kwargs,
    ):
        if self.args.e2e_dialogue_evaluation:
            return self.validate_e2e_dialogues(
                data_iterator, task, eval_dir, output_predictions_only, original_order, disable_progbar
            )
        else:
            return self.validate_batch(
                data_iterator,
                task,
                output_predictions_only,
                output_confidence_features,
                original_order,
                confidence_estimators,
                disable_progbar,
            )

    def validate_batch(
        self,
        data_iterator,
        task,
        output_predictions_only=False,
        output_confidence_features=False,
        original_order=None,
        confidence_estimators=None,
        disable_progbar=True,
    ):
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
        total_loss = 0.0 if 'loss' in task.metrics else None
        output_confidence_scores = confidence_estimators is not None
        predictions = []
        raw_predictions = []
        confidence_features = []
        example_ids = []
        answers = []
        contexts = []

        if self.numericalizer._tokenizer.tgt_lang:
            tgt_lang = self.numericalizer._tokenizer.tgt_lang
        else:
            tgt_lang = self.orig_tgt_lang

        if self.numericalizer._tokenizer.src_lang:
            src_lang = self.numericalizer._tokenizer.src_lang
        else:
            src_lang = self.orig_src_lang

        date_parser = default_loader.get_locale(src_lang[:2])

        translate_return_raw_outputs = getattr(self.args, 'translate_return_raw_outputs', False)

        for batch in progress_bar(data_iterator, desc='Generating', disable=disable_progbar):
            batch_size = len(batch.example_id)
            batch_prediction = [[] for _ in range(batch_size)]
            batch_raw_prediction = [[] for _ in range(batch_size)]
            batch_confidence_features = [[] for _ in range(batch_size)]
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
            elif output_confidence_features:
                # need gold answer for confidence estimation
                batch_answer = self.numericalizer.reverse(batch.answer.value.data, 'answer')
                answers += batch_answer

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
                        'tgt_lang': tgt_lang,
                        'date_parser': date_parser,
                    }

                    if translate_return_raw_outputs:
                        partial_batch_raw_prediction_ids = partial_batch_prediction_ids

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
                    if output_confidence_features or output_confidence_scores:
                        partial_batch_confidence_features = self.confidence_features(
                            batch=batch, predictions=partial_batch_prediction_ids, mc_dropout_num=self.args.mc_dropout_num
                        )
                    partial_batch_prediction = self.numericalizer.reverse(partial_batch_prediction_ids, 'answer')

                def get_example_index(i):
                    return (i // self.args.num_outputs[hyperparameter_idx]) % batch_size

                if translate_return_raw_outputs:
                    partial_batch_raw_prediction = self.numericalizer.reverse(partial_batch_raw_prediction_ids, 'answer')
                    for i in range(len(partial_batch_prediction)):
                        partial_batch_raw_prediction[i] = task.postprocess_prediction(
                            batch_example_ids[get_example_index(i)], partial_batch_raw_prediction[i]
                        )
                    for i in range(len(partial_batch_prediction)):
                        batch_raw_prediction[get_example_index(i)].append(partial_batch_raw_prediction[i])

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
            raw_predictions += batch_raw_prediction

        if total_loss is not None:
            total_loss /= len(example_ids)

        if original_order is not None:
            # sort back to the original order
            original_order, example_ids, predictions, raw_predictions, answers, contexts, confidence_features = [
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
                                    confidence_features,
                                )
                            )
                        )
                    )
                )
            ]

        if getattr(self.args, 'translate_example_split', False):
            # stitch sentences back together
            example_ids, predictions, raw_predictions, answers, contexts, confidence_features = merge_translated_sentences(
                example_ids,
                predictions,
                raw_predictions,
                answers,
                contexts,
                confidence_features,
                self.numericalizer._tokenizer.src_lang,
                self.numericalizer._tokenizer.tgt_lang,
            )

        if getattr(self.args, 'translate_only_entities', False):
            # stitch entities back together
            example_ids, predictions, raw_predictions, answers, contexts, confidence_features = merge_translated_sentences(
                example_ids,
                predictions,
                raw_predictions,
                answers,
                contexts,
                confidence_features,
                self.numericalizer._tokenizer.src_lang,
                self.numericalizer._tokenizer.tgt_lang,
                is_entities=True,
            )

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
        if output_confidence_features:
            output.confidence_features = confidence_features
            if self.args.override_confidence_labels:
                for i, example in enumerate(confidence_features):
                    for confidence in example:
                        confidence.label = answers[i] == self.args.override_confidence_labels
        if output_confidence_scores:
            output.confidence_scores = []
            for estimator in confidence_estimators:
                confidence_scores = estimator.estimate(confidence_features)
                output.confidence_scores.append(confidence_scores)
        if translate_return_raw_outputs:
            output.raw_predictions = raw_predictions

        return output

    def validate_e2e_dialogues(
        self, data_iterator, task, eval_dir=None, output_predictions_only=False, original_order=None, disable_progbar=True
    ):
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
        dataset_class = getattr(dialogues, task.dataset_name)
        dataset = dataset_class()

        # TODO: handle multiple responses
        hyperparameter_idx = 0

        device = self.device
        args = self.args

        if self.numericalizer._tokenizer.src_lang:
            src_lang = self.numericalizer._tokenizer.src_lang
        else:
            src_lang = self.orig_src_lang

        special_tokens = self.numericalizer._tokenizer.all_special_tokens

        e2e_dialogue_preds = dict()
        predictions = []
        example_ids = []
        answers = []
        contexts = []

        cur_dial_id = ''

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
                    partial_batch_prediction, knowledge=knowledge, lang=src_lang[:2]
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
                state_to_record = copy.deepcopy(dialogue_state)
                state_to_record = {dataset.domain2api_name(k): v for k, v in state_to_record.items()}
                e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["state"] = state_to_record
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
            confidence_estimator: if provided, will use it to calculate and output confidence scores
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

        # TODO: handle multiple responses
        hyperparameter_idx = 0

        device = self.device
        args = self.args

        if self.numericalizer._tokenizer.src_lang:
            src_lang = self.numericalizer._tokenizer.src_lang
        else:
            src_lang = self.orig_src_lang

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
                        print(colored('SYSTEM: ' + f'{INIT_SYS_MESSAGE[src_lang[:2]]}', 'red', attrs=['bold']))

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
                        partial_batch_prediction, knowledge=knowledge, lang=src_lang[:2]
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
                    state_to_record = copy.deepcopy(dialogue_state)
                    state_to_record = {dataset.domain2api_name(k): v for k, v in state_to_record.items()}
                    e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["state"] = state_to_record
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


# TransformerForSequenceClassification and TransformerForTokenClassification will inherit from this model
class GenieModelForClassification(GenieModel):
    def _init_common(self, args, tasks, **kwargs):
        self.args = args
        num_labels = 0
        if args.num_labels is not None:
            num_labels = args.num_labels
        else:
            for task in tasks:
                # if having multiple tasks choose max num_labels
                if hasattr(task, 'num_labels'):
                    num_labels = max(num_labels, task.num_labels)

        config = AutoConfig.from_pretrained(
            args.pretrained_model, cache_dir=args.embeddings, num_labels=num_labels, finetuning_task='ned'
        )
        super().__init__(config)

        if hasattr(config, 'd_model'):
            args.dimension = config.d_model
        else:
            args.dimension = config.hidden_size

        self.src_lang, self.tgt_lang = adjust_language_code(
            config, args.pretrained_model, kwargs.get('src_lang', 'en'), kwargs.get('tgt_lang', 'en')
        )

    def add_new_vocab_from_data(self, tasks, resize_decoder=False):
        super().add_new_vocab_from_data(tasks, resize_decoder)
        self.model.resize_token_embeddings(self.numericalizer.num_tokens)

    def forward(self, *input, **kwargs):
        if self.training:
            batch = input[0]
            outputs = self.model(
                batch.context.value,
                labels=batch.answer.value,
                attention_mask=(batch.context.value != self.numericalizer.pad_id),
            )
            return outputs
        else:
            return self.model(**kwargs)

    def validate(self, data_iterator, task, original_order=None, disable_progbar=True, **kwargs):
        total_loss = 0.0
        all_example_ids = []
        all_answers = []
        all_contexts = []
        all_predictions = []

        for batch in progress_bar(data_iterator, desc='Generating', disable=disable_progbar):
            batch_example_ids = batch.example_id

            batch_context = self.numericalizer.reverse(batch.context.value.data, 'context')

            all_example_ids += batch_example_ids

            # pass labels to get loss
            output = self.forward(
                input_ids=batch.context.value,
                attention_mask=(batch.context.value != self.numericalizer.pad_id),
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
                    if l_ == self.numericalizer.answer_pad_id:
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

        output = ValidationOutput(
            loss=total_loss,
            example_ids=all_example_ids,
            contexts=all_contexts,
            answers=all_answers,
            predictions=all_predictions,
        )

        return output
