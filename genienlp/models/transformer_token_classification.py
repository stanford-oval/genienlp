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

from transformers import AutoModelForTokenClassification, AutoConfig, MBartTokenizer, MBartTokenizerFast

from ..models.base import GenieModel
from ..util import adjust_language_code
from . import TransformerSeq2Seq
from ..data_utils.numericalizer import TransformerNumericalizer

logger = logging.getLogger(__name__)


class TransformerForTokenClassification(TransformerSeq2Seq, GenieModel):
    def __init__(self, config=None, *inputs, args, tasks, vocab_sets, save_directory=None, **kwargs):
    
        num_labels = 0
        if args.num_labels is not None:
            num_labels = args.num_labels
        else:
            for task in tasks:
                # if having multiple tasks choose max num_labels
                if hasattr(task, 'num_labels'):
                    num_labels = max(num_labels, task.num_labels)
        
        config = AutoConfig.from_pretrained(args.pretrained_model, cache_dir=args.embeddings, num_labels=num_labels, finetuning_task='ned')
        GenieModel.__init__(self, config)
        self.args = args
        if hasattr(config, 'd_model'):
            args.dimension = config.d_model
        else:
            args.dimension = config.hidden_size
        
        self._is_bart_large = self.args.pretrained_model == 'facebook/bart-large'
        self._is_mbart = 'mbart' in self.args.pretrained_model
        self._is_mbart50 = self._is_mbart and '-50-' in self.args.pretrained_model

        self.src_lang, self.tgt_lang = adjust_language_code(config, args.pretrained_model,
                                                            kwargs.get('src_lang', 'en'), kwargs.get('tgt_lang', 'en'))

        if save_directory is not None:
            self.model = AutoModelForTokenClassification.from_config(config)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(self.args.pretrained_model,
                                                                         cache_dir=self.args.embeddings,
                                                                         config=config)

        self.numericalizer = TransformerNumericalizer(self.args.pretrained_model, args, max_generative_vocab=None)

        self.numericalizer.get_tokenizer(save_directory, config, self.src_lang, self.tgt_lang)

        self.init_vocab_from_data(vocab_sets, tasks, save_directory)
        self.model.resize_token_embeddings(self.numericalizer.num_tokens)

        # set decoder_start_token_id
        # recommended by huggingface
        # TODO check if it's actually useful
        if self.model.config.decoder_start_token_id is None and isinstance(self.numericalizer._tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(self.numericalizer._tokenizer, MBartTokenizer):
                self.model.config.decoder_start_token_id = self.numericalizer._tokenizer.lang_code_to_id[self.tgt_lang]
            else:
                self.model.config.decoder_start_token_id = self.numericalizer._tokenizer.convert_tokens_to_ids(self.tgt_lang)

        # if self.model.config.decoder_start_token_id is None:
        #     raise ValueError("Make sure that decoder_start_token_id for the model is defined")


    def forward(self, *input, **kwargs):
        if self.training:
            batch = input[0]
            outputs = self.model(batch.context.value, labels=batch.answer.value, attention_mask=(batch.context.value!=self.numericalizer.pad_id))
            return outputs
        else:
            return self.model(**kwargs)
