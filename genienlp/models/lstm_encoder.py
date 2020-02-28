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

from torch import nn

from .common import CombinedEmbedding, PackedLSTM


class BiLSTMEncoder(nn.Module):
    def __init__(self, numericalizer, args, encoder_embeddings):
        super().__init__()
        self.args = args
        self.pad_idx = numericalizer.pad_id

        if sum(emb.dim for emb in encoder_embeddings) != args.dimension:
            raise ValueError('Hidden dimension must be equal to the sum of the embedding sizes to use IdentityEncoder')

        def dp(args):
            return args.dropout_ratio if args.rnn_layers > 1 else 0.

        self.encoder_embeddings = CombinedEmbedding(numericalizer, encoder_embeddings, args.dimension,
                                                    trained_dimension=0,
                                                    project=False,
                                                    finetune_pretrained=args.train_encoder_embeddings)

        self.bilstm_context = PackedLSTM(args.dimension, args.dimension,
                                         batch_first=True, bidirectional=True, num_layers=args.rnn_layers,
                                         dropout=dp(args))

        self.bilstm_question = PackedLSTM(args.dimension, args.dimension,
                                          batch_first=True, bidirectional=True, num_layers=args.rnn_layers,
                                          dropout=dp(args))

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

        final_context, context_rnn_state = self.bilstm_context(final_context, context_lengths)
        context_rnn_state = tuple(self.reshape_rnn_state(x) for x in context_rnn_state)

        final_question, question_rnn_state = self.bilstm_question(final_question, question_lengths)
        question_rnn_state = tuple(self.reshape_rnn_state(x) for x in question_rnn_state)

        return self_attended_context, final_context, context_rnn_state, final_question, question_rnn_state

    def reshape_rnn_state(self, h):
        # h is (num_layers * num_directions, batch, hidden_size)
        # we reshape to (num_layers, num_directions, batch, hidden_size)
        # transpose to (num_layers, batch, num_directions, hidden_size)
        # reshape to (num_layers, batch, num_directions * hidden_size)
        # also note that hidden_size is half the value of args.dimension

        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
            .transpose(1, 2).contiguous() \
            .view(h.size(0) // 2, h.size(1), h.size(2) * 2).contiguous()
