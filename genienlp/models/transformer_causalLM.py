#
# Copyright (c) 2021 The Board of Trustees of the Leland Stanford Junior University
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

from transformers import AutoConfig, AutoModelForCausalLM

from ..data_utils.numericalizer import TransformerNumericalizer
from ..models.base import GenieModelForGeneration
from ..util import adjust_language_code

logger = logging.getLogger(__name__)


class TransformerForCausalLM(GenieModelForGeneration):
    def __init__(self, config=None, *inputs, args, tasks, vocab_sets, save_directory=None, **kwargs):

        config = AutoConfig.from_pretrained(args.pretrained_model, cache_dir=args.embeddings)
        super().__init__(config)
        self.args = args
        if hasattr(config, 'd_model'):
            args.dimension = config.d_model
        else:
            args.dimension = config.hidden_size

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
            self.model = AutoModelForCausalLM.from_config(config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.pretrained_model, cache_dir=self.args.embeddings, config=config
            )

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

        self.model.resize_token_embeddings(self.numericalizer.num_tokens)

        self.numericalizer.answer_pad_id = -100

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
