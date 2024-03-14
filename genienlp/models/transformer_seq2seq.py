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
import logging
from typing import List

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, MBartTokenizer, MBartTokenizerFast

from ..data_utils.numericalizer import TransformerNumericalizer
from ..model_utils.transformers_utils import MULTILINGUAL_TOKENIZERS
from ..util import adjust_language_code
from .base import GenieModelForGeneration
from .common import LabelSmoothingCrossEntropy

logger = logging.getLogger(__name__)


class TransformerSeq2Seq(GenieModelForGeneration):
    def __init__(self, config=None, *inputs, args, tasks, vocab_sets, save_directory=None, **kwargs):
        """
        If `save_directory` is None, will initialize a new model and numericalizer, otherwise, will load them from `save_directory`
        """
        config = AutoConfig.from_pretrained(args.pretrained_model, cache_dir=args.embeddings)
        super().__init__(config)
        self.args = args
        args.dimension = self.config.d_model
        self._is_bart_large = self.args.pretrained_model == 'facebook/bart-large'

        # tasks is not passed during initialization only in server mode
        # call this function after task is recognized
        if tasks:
            self.set_generation_output_options(tasks)

        # only used for Marian models. adjusted language codes passed to numericalizer will be None for models trained on single langauge pairs
        self.orig_src_lang, self.orig_tgt_lang = kwargs.get('src_lang', 'en'), kwargs.get('tgt_lang', 'en')
        self.src_lang, self.tgt_lang = adjust_language_code(
            self.config, args.pretrained_model, kwargs.get('src_lang', 'en'), kwargs.get('tgt_lang', 'en')
        )

        if save_directory is not None:
            self.model = AutoModelForSeq2SeqLM.from_config(self.config)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.pretrained_model, cache_dir=self.args.embeddings)

        self.numericalizer = TransformerNumericalizer(
            self.args.pretrained_model,
            args,
            max_generative_vocab=None,
            save_dir=save_directory,
            config=self.config,
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

        if self._is_bart_large:
            self.config.forced_bos_token_id = None

    def forward(self, *input, **kwargs):
        if self.training or kwargs.get('train', False):
            batch = input[0]

            answer = batch.answer.value
            answer_length = batch.answer.length
            if self._is_bart_large:
                # remove BOS from the answer to BART-Large because BART-Large was not trained to predict BOS
                # (unlike BART-Base or mBART)
                #
                # NOTE: this change for some reason does not change the outputs of fine-tuned bart-large models
                # like `stanford-oval/paraphaser-bart-large`
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
