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

from .common import CombinedEmbedding, LayerNorm, LinearFeedforward


class IdentityEncoder(nn.Module):
    def __init__(self, numericalizer, args, context_embeddings, question_embeddings):
        super().__init__()
        self.args = args
        self.pad_idx = numericalizer.pad_id

        self.encoder_embeddings = CombinedEmbedding(numericalizer, context_embeddings, args.dimension,
                                                    trained_dimension=0,
                                                    project=False,
                                                    finetune_pretrained=args.train_context_embeddings)

        if self.args.rnn_layers > 0 and self.args.rnn_dimension != self.args.dimension:
            self.dropout = nn.Dropout(args.dropout_ratio)
            self.projection = nn.Linear(self.encoder_embeddings.dimension, self.args.rnn_dimension, bias=False)
        else:
            self.dropout = None
            self.projection = None

        if self.args.rnn_layers > 0 and self.args.rnn_zero_state == 'average':
            self.pool = LinearFeedforward(args.dimension, args.dimension, 2 * args.rnn_dimension * args.rnn_layers,
                                          dropout=args.dropout_ratio)
            self.norm = LayerNorm(2 * args.rnn_dimension * args.rnn_layers)
        else:
            self.pool = None
            self.norm = None

    def set_train_context_embeddings(self, trainable):
        self.encoder_embeddings.set_trainable(trainable)

    def set_train_question_embeddings(self, trainable):
        pass

    def forward(self, batch):
        context, context_lengths = batch.context.value, batch.context.length
        question, question_lengths = batch.question.value, batch.question.length

        context_padding = context.data == self.pad_idx
        question_padding = question.data == self.pad_idx

        context_embedded = self.encoder_embeddings(context, padding=context_padding)
        question_embedded = self.encoder_embeddings(question, padding=question_padding)

        # pick the top-most N transformer layers to pass to the decoder for cross-attention
        # (add 1 to account for the embedding layer - the decoder will drop it later)
        self_attended_context = context_embedded.all_layers[-(self.args.transformer_layers + 1):]
        final_context = context_embedded.last_layer
        final_question = question_embedded.last_layer

        if self.projection is not None:
            final_context = self.dropout(final_context)
            final_context = self.projection(final_context)

            final_question = self.dropout(final_question)
            final_question = self.projection(final_question)

        context_rnn_state = None
        question_rnn_state = None
        if self.args.rnn_layers > 0:
            if self.args.rnn_zero_state == 'zero':
                batch_size = context.size(0)

                zero = torch.zeros(self.args.rnn_layers, batch_size, self.args.rnn_dimension,
                                   dtype=torch.float, requires_grad=False, device=context.device)
                context_rnn_state = (zero, zero)
                question_rnn_state = (zero, zero)
            else:
                assert self.args.rnn_zero_state == 'average'
                batch_size = context.size(0)

                masked_final_context = context_embedded.last_layer.masked_fill(context_padding.unsqueeze(2), 0)
                summed_context = torch.sum(masked_final_context, dim=1)
                average_context = summed_context / context_lengths.unsqueeze(1)

                packed_rnn_state = self.norm(self.pool(average_context))

                # packed_rnn_state is (batch, 2 * rnn_layers * rnn_dim)
                packed_rnn_state = packed_rnn_state.reshape(batch_size, 2, self.args.rnn_layers,
                                                            self.args.rnn_dimension)
                # transpose to (2, batch, rnn_layers, rnn_dimension)
                packed_rnn_state = packed_rnn_state.transpose(0, 1)
                # transpose to (2, rnn_layers, batch, rnn_dimension)
                packed_rnn_state = packed_rnn_state.transpose(1, 2)
                # convert to a tuple of two (rnn_layers, batch, rnn_dimension) tensors
                packed_rnn_state = packed_rnn_state.chunk(2, dim=0)
                context_rnn_state = (packed_rnn_state[0].squeeze(0), packed_rnn_state[1].squeeze(0))

        return self_attended_context, final_context, context_rnn_state, final_question, question_rnn_state
