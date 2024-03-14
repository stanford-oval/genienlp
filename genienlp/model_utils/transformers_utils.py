# Parts of this file were adopted from https://github.com/huggingface/transformers
#
# Copyright 2021 The Board of Trustees of the Leland Stanford Junior University
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
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
import random

import torch
import torch.nn as nn
from transformers import M2M100Tokenizer, MBart50Tokenizer, MBart50TokenizerFast, MBartTokenizerFast, XLMRobertaConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel
from transformers.models.gpt2 import tokenization_gpt2
from transformers.models.marian.convert_marian_tatoeba_to_pytorch import GROUP_MEMBERS
from transformers.models.marian.convert_marian_to_pytorch import GROUPS
from transformers.models.mbart.tokenization_mbart import MBartTokenizer
from transformers.models.roberta.modeling_roberta import (
    RobertaEncoder,
    RobertaPooler,
    RobertaPreTrainedModel,
    create_position_ids_from_input_ids,
)
from transformers.models.t5 import tokenization_t5

logger = logging.getLogger(__name__)


MT5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'google/mt5-{}'.format(v): "https://s3.amazonaws.com/models.huggingface.co/bert/google/mt5-{}/config.json".format(v)
    for v in ['small', 'base', 'large', 'xl', 'xxl']
}

MULTILINGUAL_TOKENIZERS = (MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer)


MARIAN_GROUPS = {item[1]: set(item[0].split('+')) for item in GROUPS}
MARIAN_TATOEBA_GROUPS = {k: set(v[1]) for k, v in GROUP_MEMBERS.items()}

MARIAN_GROUP_MEMBERS = {**MARIAN_GROUPS, **MARIAN_TATOEBA_GROUPS}


MODEL_PARALLEL_SUPPORTED_MODELS = (
    list(tokenization_gpt2.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES.keys())
    + list(tokenization_t5.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES.keys())
    + list(MT5_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
)

###############


class GenieMBartTokenizer(MBartTokenizer):
    '''
    MBartTokenizer with the fix for off-by-one error during generation: https://github.com/huggingface/transformers/issues/5755
    '''

    def __init__(self, *args, tokenizer_file=None, **kwargs):
        super().__init__(*args, tokenizer_file=tokenizer_file, **kwargs)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. Prefix [bos_token_id], suffix =[eos_token_id]."""
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. Prefix [tgt_lang_code], suffix =[eos_token_id]."""
        self.cur_lang_code = self.lang_code_to_id[lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]
