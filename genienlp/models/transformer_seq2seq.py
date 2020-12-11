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
from transformers import AutoModelForSeq2SeqLM, AutoConfig

from ..data_utils.numericalizer import TransformerNumericalizer
from .base import GenieModel

logger = logging.getLogger(__name__)


class TransformerSeq2Seq(GenieModel):
    def __init__(self, config=None, *inputs, args, tasks, vocab_sets, save_directory=None, **kwargs):
        config = AutoConfig.from_pretrained(args.pretrained_model, cache_dir=args.embeddings)
        super().__init__(config)
        self.args = args
        args.dimension = config.d_model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.pretrained_model,
                                                           cache_dir=self.args.embeddings)
        self.numericalizer = TransformerNumericalizer(self.args.pretrained_model, max_generative_vocab=None,
                                                      preprocess_special_tokens=args.preprocess_special_tokens)
        self.init_vocab_from_data(vocab_sets, tasks, save_directory)
        self.model.resize_token_embeddings(self.numericalizer.num_tokens)

    def add_new_vocab_from_data(self, tasks, resize_decoder=False):
        super().add_new_vocab_from_data(tasks, resize_decoder)
        self.model.resize_token_embeddings(self.numericalizer.num_tokens)

    def forward(self, *input, **kwargs):
        if self.training:
            batch = input[0]
            pad = self.numericalizer._tokenizer.pad_token_id
            source_ids, source_mask, y = batch.context.value, batch.context.value != pad, batch.answer.value
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone()
            labels[y[:, 1:] == pad] = -100
            return self.model(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids, labels=labels)
        else:
            return self.model(**kwargs)

    def generate(self,
                 batch,
                 max_output_length,
                 num_outputs,
                 temperature,
                 repetition_penalty,
                 top_k,
                 top_p,
                 num_beams,
                 no_repeat_ngram_size,
                 do_sample
                 ):

        input_ids = batch.context.value
        # TODO attention_mask
        generated = self.model.generate(input_ids=input_ids,
                                        max_length=max_output_length,
                                        min_length=2,  # generate at least one token after BOS
                                        bos_token_id=self.numericalizer._tokenizer.bos_token_id,
                                        pad_token_id=self.numericalizer._tokenizer.pad_token_id,
                                        early_stopping=True,
                                        num_return_sequences=num_outputs,
                                        repetition_penalty=repetition_penalty,
                                        temperature=temperature,
                                        eos_token_id=self.numericalizer._tokenizer.eos_token_id,
                                        top_k=top_k,
                                        top_p=top_p,
                                        num_beams=num_beams,
                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                        do_sample=do_sample,
                                        )

        return generated
