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
from typing import List

import torch
import ujson
from dateparser.languages import default_loader
from dialogues import Bitod
from transformers import AutoConfig, AutoModelForSeq2SeqLM, MarianTokenizer, MBartTokenizer, MBartTokenizerFast

from ..data_utils.example import NumericalizedExamples, SequentialField
from ..data_utils.numericalizer import TransformerNumericalizer
from ..data_utils.progbar import progress_bar
from ..model_utils.transformers_utils import MULTILINGUAL_TOKENIZERS
from ..util import (
    ConfidenceFeatures,
    GenerationOutput,
    adjust_language_code,
    merge_translated_sentences,
    replace_capturing_group,
)
from .base import GenieModel
from .common import LabelSmoothingCrossEntropy

logger = logging.getLogger(__name__)


class TransformerSeq2Seq(GenieModel):
    def __init__(self, config=None, *inputs, args, tasks, vocab_sets, save_directory=None, **kwargs):
        """
        If `save_directory` is None, will initialize a new model and numericalizer, otherwise, will load them from `save_directory`
        """
        config = AutoConfig.from_pretrained(args.pretrained_model, cache_dir=args.embeddings)
        self.config = config
        super().__init__(config)
        self.args = args
        args.dimension = config.d_model
        self._is_bart_large = self.args.pretrained_model == 'facebook/bart-large'

        # tasks is not passed during initialization only in server mode
        # call this function after task is recognized
        if tasks:
            self.set_generation_output_options(tasks)

        # only used for Marian models. adjusted language codes passed to numericalizer will be None for models trained on single langauge pairs
        self.orig_src_lang, self.orig_tgt_lang = kwargs.get('src_lang', 'en'), kwargs.get('tgt_lang', 'en')
        self.src_lang, self.tgt_lang = adjust_language_code(
            config, args.pretrained_model, kwargs.get('src_lang', 'en'), kwargs.get('tgt_lang', 'en')
        )

        if save_directory is not None:
            self.model = AutoModelForSeq2SeqLM.from_config(config)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.pretrained_model, cache_dir=self.args.embeddings)

        self.numericalizer = TransformerNumericalizer(
            self.args.pretrained_model,
            args,
            max_generative_vocab=None,
            save_dir=save_directory,
            config=config,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            vocab_sets=vocab_sets,
            tasks=tasks,
        )

        self.update_language_dependent_configs(self.tgt_lang)
        self.model.resize_token_embeddings(self.numericalizer.num_tokens)

        if args.dropper_ratio > 0:
            # lazy import since dropper is an optional dependency
            from loss_dropper import LossDropper

            self.dropper = LossDropper(dropc=args.dropper_ratio, min_count=args.dropper_min_count)
        else:
            self.dropper = None

        self.criterion = LabelSmoothingCrossEntropy(args.label_smoothing)

    def add_new_vocab_from_data(self, tasks, resize_decoder=False):
        super().add_new_vocab_from_data(tasks, resize_decoder)
        self.model.resize_token_embeddings(self.numericalizer.num_tokens)

    def update_language_dependent_configs(self, tgt_lang):
        # set decoder_start_token_id for mbart
        if self.config.decoder_start_token_id is None and isinstance(
            self.numericalizer._tokenizer, (MBartTokenizer, MBartTokenizerFast)
        ):
            if isinstance(self.numericalizer._tokenizer, MBartTokenizer):
                self.config.decoder_start_token_id = self.numericalizer._tokenizer.lang_code_to_id[tgt_lang]
            else:
                self.config.decoder_start_token_id = self.numericalizer._tokenizer.convert_tokens_to_ids(tgt_lang)

        # check decoder_start_token_id is set
        if self.config.decoder_start_token_id is None:
            raise ValueError("Make sure that decoder_start_token_id for the model is defined")

        # set forced_bos_token_id for certain multilingual models
        if isinstance(self.numericalizer._tokenizer, MULTILINGUAL_TOKENIZERS):
            forced_bos_token_id = self.numericalizer._tokenizer.lang_code_to_id[tgt_lang]
            self.config.forced_bos_token_id = forced_bos_token_id

    def forward(self, *input, **kwargs):
        if self.training or kwargs.get('train', False):
            batch = input[0]

            answer = batch.answer.value
            answer_length = batch.answer.length
            if self._is_bart_large:
                # remove BOS from the answer to BART-Large because BART-Large was not trained to predict BOS
                # (unlike BART-Base or mBART)
                #
                # NOTE: various people at Huggingface and elsewhere have tried to conclusively ascertain
                # whether BOS should be there or not, and the answer seems to be that BOS should not be there
                # at all, either in input or in the output
                # but empirically, BOS in the input works slightly better, perhaps because our sentences start
                # with a lowercase letter, so we leave it
                answer = answer[:, 1:].contiguous()
                answer_length = answer_length - 1

            # this has several differences compared to what `transformers` Seq2Seq models do:
            # (1) pad tokens are ignored in all loss calculations
            # (2) loss is averaged over sequence lengths first, then over the batch size. This way,
            # longer sequences in the batch do not drown shorter sequences.
            # (3) if `args.dropper_ratio > 0.0`, will perform Loss Truncation
            # (4) if `args.label_smoothing > 0.0`, will add label smoothing term to loss
            outputs = self.model(
                batch.context.value,
                labels=answer,
                attention_mask=(batch.context.value != self.numericalizer.pad_id),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            batch_size, vocab_size = outputs.logits.shape[0], outputs.logits.shape[2]
            loss = self.criterion(
                outputs.logits.view(-1, vocab_size), target=answer.view(-1), ignore_index=self.numericalizer.pad_id
            )
            loss = loss.view(batch_size, -1)  # (batch_size, sequence_length)
            loss = loss.sum(dim=1) / answer_length  # accounts for the case where BOS is removed
            if self.dropper is not None:
                dropper_mask = self.dropper(loss)
                loss = loss * dropper_mask
            loss = loss.mean()  # average over the batch size
            outputs.loss = loss  # replace the loss calculated by `transformers` with the new loss
            return outputs
        else:
            return self.model(**kwargs)

    def generate(
        self,
        batch,
        max_output_length,
        min_output_length,
        num_outputs,
        temperature,
        repetition_penalty,
        top_k,
        top_p,
        num_beams,
        num_beam_groups,
        diversity_penalty,
        no_repeat_ngram_size,
        do_sample,
    ):

        input_ids = batch.context.value

        # when attention_mask is not provided to generate(), it will default to masking pad tokens, which is the correct thing
        generated = self.model.generate(
            input_ids=input_ids,
            max_length=max_output_length,
            min_length=min_output_length,
            bos_token_id=self.numericalizer.init_id,
            pad_token_id=self.numericalizer.pad_id,
            early_stopping=False,
            num_return_sequences=num_outputs,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            eos_token_id=self.numericalizer.eos_id,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            output_scores=self._output_scores,
            output_attentions=self._output_attentions,
            output_hidden_states=self._output_hidden_states,
            return_dict_in_generate=True,
        )

        return generated

    def validate(
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
            tgt_lang = self.model.orig_tgt_lang

        if self.numericalizer._tokenizer.src_lang:
            src_lang = self.numericalizer._tokenizer.src_lang
        else:
            src_lang = self.model.orig_src_lang

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
                        'self.numericalizer': self.numericalizer,
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
                        partial_batch_confidence_features = self.model.confidence_features(
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

        output = GenerationOutput(loss=total_loss)

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
        self, data_iterator, task, eval_dir, output_predictions_only=False, original_order=None, disable_progbar=True
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

        dataset = Bitod()
        e2e_dialogue_preds = dict()

        predictions = []
        example_ids = []
        answers = []
        contexts = []

        # TODO: handle multiple responses
        hyperparameter_idx = 0

        cur_dial_id = ''
        knowledge = None

        device = self.device
        args = self.args

        special_tokens = self.numericalizer._tokenizer.all_special_tokens

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
                # new_state_text = 'null'
                knowledge = defaultdict(dict)
                new_knowledge_text = 'null'
                new_actions_text = 'null'
                active_api = None
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

                batch_context.append(self.numericalizer._tokenizer.convert_tokens_to_string(text))

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

            tokenized_contexts = self.numericalizer.encode_batch([input_text], field_name='context', features=None)[0]

            numericalized_turn = NumericalizedExamples(
                example_id=[turn.example_id[0]],
                context=SequentialField(
                    value=torch.tensor([tokenized_contexts.value], device=device),
                    length=torch.tensor([tokenized_contexts.length], device=device),
                    limited=torch.tensor([tokenized_contexts.limited], device=device),
                    feature=None,
                ),
                answer=SequentialField(value=None, length=None, limited=None, feature=None),
            )

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
                    partial_batch_prediction, knowledge, lang=self.numericalizer._tokenizer.src_lang[:2]
                )

            partial_batch_prediction = task.postprocess_prediction(batch_example_ids[0], partial_batch_prediction)

            # put them into the right array
            batch_prediction.append([partial_batch_prediction])

            predictions += batch_prediction

            if train_target == 'dst':
                # update dialogue_state
                lev = predictions[-1][0].strip()
                state_update = dataset.span2state(lev)
                if state_update:
                    active_api = list(state_update.keys())[-1]
                dataset.update_state(state_update, dialogue_state)

                #### save latest state
                state_to_record = copy.deepcopy(dialogue_state)
                state_to_record = {dataset.domain2api_name(k): v for k, v in state_to_record.items()}
                e2e_dialogue_preds[dial_id]["turns"][str(turn_id)]["state"] = state_to_record
                ####

            elif train_target == 'api':
                if dataset.do_knowledge_reset(active_api):
                    new_knowledge_text = "null"
                    knowledge = defaultdict(dict)

                do_api_call = predictions[-1][0].strip()

                if do_api_call == 'yes':
                    # make api call
                    api_name = active_api
                    if api_name in dialogue_state:
                        constraints, new_knowledge_text = dataset.make_api_call(
                            dialogue_state, knowledge, api_name, self.numericalizer._tokenizer.src_lang, dial_id, turn_id
                        )
                        #### save latest api constraints
                        e2e_dialogue_preds[dial_id]["API"][dataset.domain2api_name(api_name)] = copy.deepcopy(constraints)
                        ####

                elif do_api_call == 'no':
                    # do nothing
                    pass
                else:
                    logger.error(
                        f'API call should be either yes or no but got {do_api_call}. Seems model is not trained for enough steps. For now we assume it\'s a no'
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

        with open(os.path.join(eval_dir, 'e2e_dialogue_preds.json'), 'w') as fout:
            ujson.dump(e2e_dialogue_preds, fout, indent=2, ensure_ascii=False)

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
            output.example_ids, output.predictions, output.answers, output.contexts = (
                example_ids,
                predictions,
                answers,
                contexts,
            )

        return output

    def confidence_features(self, batch, predictions, mc_dropout_num=0) -> List[ConfidenceFeatures]:
        """
        predictions: Tensor of shape (batch_size, output_length)
        mc_droput_num: number of Monte Carlo samples used for the MC Dropout method. 0 disables MC dropout.
        """
        batch_size = predictions.shape[0]
        repetition_factor = batch_size // batch.context.value.shape[0]
        input_ids = batch.context.value.repeat_interleave(
            repetition_factor, dim=0
        )  # repeat to account for multiple predictions per input

        prediction_lengths = self.get_length(predictions)

        pad_token_id = self.numericalizer.pad_id
        attention_mask = self.model._prepare_attention_mask_for_generation(
            inputs=input_ids, pad_token_id=pad_token_id, eos_token_id=self.numericalizer.eos_id
        )
        truncated_predictions = predictions[:, 1:]  # remove the BOS token since it is not actually being generated

        assert not self.training, 'Model should be in eval() mode before generation can start.'

        batch_nodrop_logits = []
        batch_nodrop_probs = []
        batch_nodrop_entropies = []
        # batch_nodrop_top1_probs = []
        # batch_nodrop_top1_idx = []
        # batch_nodrop_top2_probs = []
        # batch_nodrop_top2_idx = []
        outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=predictions,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        nodrop_logits = outputs.logits[:, :-1, :]  # remove the last probability distribution which is for the token after EOS
        for i in range(batch_size):
            batch_nodrop_logits.append(
                nodrop_logits[i].gather(dim=1, index=truncated_predictions[i].view(-1, 1)).view(-1)[: prediction_lengths[i]]
            )
            probs = torch.softmax(nodrop_logits[i], dim=1)
            batch_nodrop_probs.append(
                probs.gather(dim=1, index=truncated_predictions[i].view(-1, 1)).view(-1)[: prediction_lengths[i]]
            )
            batch_nodrop_entropies.append(-torch.sum(torch.log(probs) * probs, dim=1)[: prediction_lengths[i]])
            # sorted_probs = probs.sort(dim=1)
            # batch_nodrop_top1_probs.append(sorted_probs.values[:, -1][:prediction_lengths[i]])
            # batch_nodrop_top2_probs.append(sorted_probs.values[:, -2][:prediction_lengths[i]])
            # batch_nodrop_top1_idx.append(sorted_probs.indices[:, -1][:prediction_lengths[i]])
            # batch_nodrop_top2_idx.append(sorted_probs.indices[:, -2][:prediction_lengths[i]])

        # activate dropout layers
        self.train()

        batch_drop_logits = [[] for _ in range(batch_size)]
        batch_drop_probs = [[] for _ in range(batch_size)]
        # batch_drop_top1_probs = [[] for _ in range(batch_size)]
        # batch_drop_top2_probs = [[] for _ in range(batch_size)]
        for _ in range(mc_dropout_num):
            outputs = self.model(
                input_ids=input_ids,
                decoder_input_ids=predictions,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )
            drop_logits = outputs.logits[
                :, :-1, :
            ]  # remove the last probability distribution which is for the token after EOS
            for i in range(batch_size):
                batch_drop_logits[i].append(
                    (drop_logits[i].gather(dim=1, index=truncated_predictions[i].view(-1, 1)).view(-1))[
                        : prediction_lengths[i]
                    ]
                )
                softmax = torch.softmax(drop_logits[i], dim=1)
                drop_probs = softmax.gather(dim=1, index=truncated_predictions[i].view(-1, 1)).view(-1)[
                    : prediction_lengths[i]
                ]
                batch_drop_probs[i].append(drop_probs)
                # batch_drop_top1_probs[i].append(softmax.gather(dim=1, index=batch_nodrop_top1_idx[i].view(-1, 1)).view(-1)[:prediction_lengths[i]])
                # batch_drop_top2_probs[i].append(softmax.gather(dim=1, index=batch_nodrop_top2_idx[i].view(-1, 1)).view(-1)[:prediction_lengths[i]])

        confidence_features = []
        for i in range(batch_size):
            # TODO refactor so token changes (EOS/ BOS removal, etc.) is decided based on specific tokenizer used
            if self._is_bart_large:
                prediction = predictions[i][: prediction_lengths[i] + 1]  # +1 to include EOS
            else:
                prediction = predictions[i][1 : prediction_lengths[i] + 1]  # remove token before BOS, +1 to include EOS
            confidence_features.append(
                ConfidenceFeatures(
                    drop_logits=batch_drop_logits[i] if mc_dropout_num > 0 else None,
                    drop_probs=batch_drop_probs[i] if mc_dropout_num > 0 else None,
                    #  drop_top1_probs=batch_drop_top1_probs[i] if mc_dropout_num > 0 else None,
                    #  drop_top2_probs=batch_drop_top2_probs[i] if mc_dropout_num > 0 else None,
                    gold_answer=batch.answer.value[i // repetition_factor][: batch.answer.length[i // repetition_factor]],
                    prediction=prediction,
                    nodrop_logits=batch_nodrop_logits[i][: prediction_lengths[i]],
                    nodrop_probs=batch_nodrop_probs[i][: prediction_lengths[i]],
                    #  nodrop_top1_probs=batch_nodrop_top1_probs[i][:prediction_lengths[i]],
                    #  nodrop_top2_probs=batch_nodrop_top2_probs[i][:prediction_lengths[i]],
                    nodrop_entropies=batch_nodrop_entropies[i][: prediction_lengths[i]],
                    context=batch.context.value[i // repetition_factor][: batch.context.length[i // repetition_factor]],
                )
            )

        # return the model back to its previous state
        self.eval()

        return confidence_features

    def get_length(self, prediction: torch.Tensor):
        # skip the first token, because BOS is the same as EOS for some models
        prediction = prediction[:, 1:]

        # add EOS at the end in case the prediction doesn't have any
        prediction = torch.cat(
            [
                prediction,
                torch.ones((prediction.shape[0], 1), dtype=torch.long, device=prediction.device) * self.numericalizer.eos_id,
            ],
            dim=1,
        )

        # find the index of the first eos
        first_eos_one_hot = (torch.cumsum((prediction == self.numericalizer.eos_id).long(), dim=1) == 1) & (
            prediction == self.numericalizer.eos_id
        )
        first_eos = first_eos_one_hot.nonzero(as_tuple=False)[:, 1] + 1  # +1 to account for the first token that we ignored
        return first_eos
