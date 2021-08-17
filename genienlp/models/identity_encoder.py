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

import torch
from torch import nn

from .common import LayerNorm, LinearFeedforward


class IdentityEncoder(nn.Module):
    def __init__(self, numericalizer, args, config, context_embeddings):
        super().__init__()
        self.args = args
        self.pad_idx = numericalizer.pad_id

        if args.rnn_dimension is None:
            args.rnn_dimension = config.hidden_size

        self.encoder_embeddings = context_embeddings

        if self.args.rnn_layers > 0 and self.args.rnn_dimension != config.hidden_size:
            self.dropout = nn.Dropout(args.dropout_ratio)
            self.projection = nn.Linear(config.hidden_size, self.args.rnn_dimension, bias=False)
        else:
            self.dropout = None
            self.projection = None

        if self.args.rnn_layers > 0 and self.args.rnn_zero_state in ['average', 'cls']:
            self.pool = LinearFeedforward(
                config.hidden_size, config.hidden_size, 2 * args.rnn_dimension * args.rnn_layers, dropout=args.dropout_ratio
            )
            self.norm = LayerNorm(2 * args.rnn_dimension * args.rnn_layers)
        else:
            self.pool = None
            self.norm = None

    def compute_final_embeddings(
        self,
        context,
        context_lengths,
        context_padding,
        context_entity_ids,
        context_entity_probs=None,
        context_entity_masking=None,
        entity_word_embeds_dropout=True,
    ):

        if self.args.do_ned:
            context_embedded_last_hidden_state = self.encoder_embeddings(
                context,
                entity_ids=context_entity_ids,
                entity_masking=context_entity_masking,
                entity_probs=context_entity_probs,
                entity_word_embeds_dropout=entity_word_embeds_dropout,
            ).last_hidden_state

        else:
            context_embedded_last_hidden_state = self.encoder_embeddings(
                context, attention_mask=(~context_padding).to(dtype=torch.float)
            ).last_hidden_state

        final_context = context_embedded_last_hidden_state

        if self.projection is not None:
            final_context = self.dropout(final_context)
            final_context = self.projection(final_context)

        context_rnn_state = None
        if self.args.rnn_layers > 0:
            batch_size = context.size(0)
            if self.args.rnn_zero_state == 'zero':

                zero = torch.zeros(
                    self.args.rnn_layers,
                    batch_size,
                    self.args.rnn_dimension,
                    dtype=torch.float,
                    requires_grad=False,
                    device=context.device,
                )
                context_rnn_state = (zero, zero)
            else:
                if self.args.rnn_zero_state == 'cls':
                    packed_rnn_state = self.norm(self.pool(final_context[:, 0, :]))

                else:
                    assert self.args.rnn_zero_state == 'average'
                    masked_final_context = final_context.masked_fill(context_padding.unsqueeze(2), 0)
                    summed_context = torch.sum(masked_final_context, dim=1)
                    average_context = summed_context / context_lengths.unsqueeze(1)

                    packed_rnn_state = self.norm(self.pool(average_context))

                # packed_rnn_state is (batch, 2 * rnn_layers * rnn_dim)
                packed_rnn_state = packed_rnn_state.reshape(batch_size, 2, self.args.rnn_layers, self.args.rnn_dimension)
                # transpose to (2, batch, rnn_layers, rnn_dimension)
                packed_rnn_state = packed_rnn_state.transpose(0, 1)
                # transpose to (2, rnn_layers, batch, rnn_dimension)
                packed_rnn_state = packed_rnn_state.transpose(1, 2)
                # convert to a tuple of two (rnn_layers, batch, rnn_dimension) tensors
                packed_rnn_state = packed_rnn_state.chunk(2, dim=0)
                context_rnn_state = (packed_rnn_state[0].squeeze(0), packed_rnn_state[1].squeeze(0))

        return final_context, context_rnn_state

    def forward(self, batch):
        context, context_lengths = batch.context.value, batch.context.length
        context_padding = torch.eq(context.data, self.pad_idx)

        context_entity_ids, context_entity_probs, context_entity_masking = None, None, None

        if self.args.do_ned and self.args.add_entities_to_text == 'off':
            context_entity_ids = batch.context.feature[:, :, : self.args.max_features_size].long()

            # indicates position of entities
            context_entity_masking = (context_entity_ids != 0).int()

            if self.args.entity_type_agg_method == 'weighted':
                context_entity_probs = batch.context.feature[
                    :, :, self.args.max_features_size : self.args.max_features_size + self.args.max_features_size
                ].long()

        final_context, context_rnn_state = self.compute_final_embeddings(
            context,
            context_lengths,
            context_padding,
            context_entity_ids,
            context_entity_probs,
            context_entity_masking,
            entity_word_embeds_dropout=self.args.entity_word_embeds_dropout,
        )

        return final_context, context_rnn_state
