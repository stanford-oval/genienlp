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

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, MBartTokenizer, MBartTokenizerFast

from ..data_utils.numericalizer import TransformerNumericalizer
from .base import GenieModelForGeneration

logger = logging.getLogger(__name__)


class TransformerSeq2Seq(GenieModelForGeneration):
    def __init__(self, config=None, *inputs, args, tasks, vocab_sets, save_directory=None, **kwargs):
        """
        If `save_directory` is None, will initialize a new model and numericalizer, otherwise, will load them from `save_directory`
        """
        config = AutoConfig.from_pretrained(args.pretrained_model)
        super().__init__(config)
        self.args = args
        args.dimension = self.config.d_model
        self._is_bart_large = self.args.pretrained_model == 'facebook/bart-large'

        # tasks is not passed during initialization only in server mode
        # call this function after task is recognized

        if save_directory is not None:
            self.model = AutoModelForSeq2SeqLM.from_config(self.config)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.pretrained_model)

        self.numericalizer = TransformerNumericalizer(
            self.args.pretrained_model,
            args,
            max_generative_vocab=None,
            save_dir=save_directory,
            config=self.config,
            vocab_sets=vocab_sets,
            tasks=tasks,
        )

        self.model.resize_token_embeddings(self.numericalizer.num_tokens)


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

            outputs = self.model(
                batch.context.value,
                labels=answer,
                attention_mask=(batch.context.value != self.numericalizer.pad_id),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            outputs.loss = 0 
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
                torch.ones((prediction.shape[0], 1), dtype=torch.long) * self.numericalizer.eos_id,
            ],
            dim=1,
        )

        # find the index of the first eos
        first_eos_one_hot = (torch.cumsum((prediction == self.numericalizer.eos_id).long(), dim=1) == 1) & (
            prediction == self.numericalizer.eos_id
        )
        first_eos = first_eos_one_hot.nonzero(as_tuple=False)[:, 1] + 1  # +1 to account for the first token that we ignored
        return first_eos
