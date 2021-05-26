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


###############
## NER models
###############
## BERT


class BertEmbeddingsForNER(BertEmbeddings):
    """Construct the embeddings from word, position, token_type, and entity_type embeddings."""

    def __init__(self, config, num_db_types, db_unk_id):
        super().__init__(config)
        self.num_db_types = num_db_types
        self.db_unk_id = db_unk_id
        self.pad_token_id = config.pad_token_id
        if num_db_types > 0:
            self.entity_type_embeddings = nn.Embedding(num_db_types, config.hidden_size, padding_idx=db_unk_id)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        entity_ids=None,
        entity_masking=None,
        entity_probs=None,
        entity_word_embeds_dropout=False,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # dropout word embeddings for entities
        if entity_masking is not None and entity_word_embeds_dropout > 0.0:
            if entity_word_embeds_dropout > random.random():
                input_ids = input_ids * ~(entity_masking.max(-1)[0].bool())

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if self.num_db_types > 0 and entity_ids is not None:
            # get length for unpadded types
            type_lengths = entity_masking.sum(-1)

            # avoid division by zero
            type_lengths[type_lengths == 0] = 1

            entity_type_embeddings = self.entity_type_embeddings(entity_ids)

            # weighted average
            if entity_probs is not None:
                entity_type_embeddings = entity_type_embeddings * entity_probs.unsqueeze(-1)

            # average embedding of different types
            # size (batch, length, num_types, emb_dim)
            entity_type_embeddings = entity_type_embeddings.sum(-2) / type_lengths.unsqueeze(-1)

            embeddings += entity_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelForNER(BertModel):
    """
    Subcalss of BertModel model with an additional entity type embedding layer in the bottom
    """

    def __init__(self, config, num_db_types, db_unk_id):
        super().__init__(config)

        self.embeddings = BertEmbeddingsForNER(config, num_db_types, db_unk_id)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        entity_masking=None,
        entity_probs=None,
        entity_word_embeds_dropout=False,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        embedding_output=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if embedding_output is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                entity_ids=entity_ids,
                entity_masking=entity_masking,
                entity_probs=entity_probs,
                entity_word_embeds_dropout=entity_word_embeds_dropout,
            )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


###############
## Roberta


class RobertaEmbeddingsForNER(BertEmbeddingsForNER):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing and adding entity_types.
    """

    def __init__(self, config, num_db_types, db_unk_id):
        super().__init__(config, num_db_types, db_unk_id)
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
        entity_ids=None,
        entity_masking=None,
        entity_probs=None,
        entity_word_embeds_dropout=False,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super().forward(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids,
            entity_masking=entity_masking,
            entity_probs=entity_probs,
            entity_word_embeds_dropout=entity_word_embeds_dropout,
        )

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaModelForNER(RobertaPreTrainedModel):
    def __init__(self, config, num_db_types, db_unk_id, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddingsForNER(config, num_db_types, db_unk_id)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        entity_masking=None,
        entity_probs=None,
        entity_word_embeds_dropout=False,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        embedding_output=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if embedding_output is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                entity_ids=entity_ids,
                entity_masking=entity_masking,
                entity_probs=entity_probs,
                entity_word_embeds_dropout=entity_word_embeds_dropout,
            )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class XLMRobertaModelForNER(RobertaModelForNER):
    """
    Subcalss of XLMRobertaModel model with an additional entity type embedding layer at the bottom
    """

    config_class = XLMRobertaConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_db_types, db_unk_id, add_pooling_layer=True):
        super().__init__(config, num_db_types, db_unk_id, add_pooling_layer)
