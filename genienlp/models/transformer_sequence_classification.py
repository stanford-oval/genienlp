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

from transformers import AutoModelForSequenceClassification

from ..data_utils.numericalizer import TransformerNumericalizer
from .transformer_token_classification import TransformerForTokenClassification

logger = logging.getLogger(__name__)


class TransformerForSequenceClassification(TransformerForTokenClassification):
    def __init__(self, config=None, *inputs, args, tasks, vocab_sets, save_directory=None, **kwargs):

        self._init_common(args, tasks, **kwargs)

        if save_directory is not None:
            self.model = AutoModelForSequenceClassification.from_config(self.config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args.pretrained_model, cache_dir=self.args.embeddings, config=self.config
            )

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

        self.model.resize_token_embeddings(self.numericalizer.num_tokens)

        self.numericalizer.answer_pad_id = -100
