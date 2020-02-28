#
# Copyright (c) 2018, Salesforce, Inc.
#                     The Board of Trustees of the Leland Stanford Junior University
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

from .common import CombinedEmbedding, PackedLSTM, CoattentiveLayer, TransformerEncoder


class MQANEncoder(nn.Module):
    def __init__(self, numericalizer, args, context_embeddings, question_embeddings):
        super().__init__()
        self.args = args
        self.pad_idx = numericalizer.pad_id

        self.encoder_embeddings = CombinedEmbedding(numericalizer, context_embeddings, args.dimension,
                                                    trained_dimension=0,
                                                    project=True,
                                                    finetune_pretrained=args.train_encoder_embeddings)

        def dp(args):
            return args.dropout_ratio if args.rnn_layers > 1 else 0.

        self.bilstm_before_coattention = PackedLSTM(args.dimension, args.dimension,
                                                    batch_first=True, bidirectional=True, num_layers=1, dropout=0)
        self.coattention = CoattentiveLayer(args.dimension, dropout=0.3)
        dim = 2 * args.dimension + args.dimension + args.dimension

        self.context_bilstm_after_coattention = PackedLSTM(dim, args.dimension,
                                                           batch_first=True, dropout=dp(args), bidirectional=True,
                                                           num_layers=args.rnn_layers)
        self.self_attentive_encoder_context = TransformerEncoder(args.dimension, args.transformer_heads,
                                                                 args.transformer_hidden, args.transformer_layers,
                                                                 args.dropout_ratio)
        self.bilstm_context = PackedLSTM(args.dimension, args.dimension,
                                         batch_first=True, dropout=dp(args), bidirectional=True,
                                         num_layers=args.rnn_layers)

        self.question_bilstm_after_coattention = PackedLSTM(dim, args.dimension,
                                                            batch_first=True, dropout=dp(args), bidirectional=True,
                                                            num_layers=args.rnn_layers)
        self.self_attentive_encoder_question = TransformerEncoder(args.dimension, args.transformer_heads,
                                                                  args.transformer_hidden, args.transformer_layers,
                                                                  args.dropout_ratio)
        self.bilstm_question = PackedLSTM(args.dimension, args.dimension,
                                          batch_first=True, dropout=dp(args), bidirectional=True,
                                          num_layers=args.rnn_layers)

    def set_train_context_embeddings(self, trainable):
        self.encoder_embeddings.set_trainable(trainable)

    def set_train_question_embeddings(self, trainable):
        pass

    def forward(self, batch):
        context, context_lengths = batch.context.value, batch.context.length
        question, question_lengths = batch.question.value, batch.question.length

        context_padding = context.data == self.pad_idx
        question_padding = question.data == self.pad_idx

        context_embedded = self.encoder_embeddings(context, padding=context_padding).last_layer
        question_embedded = self.encoder_embeddings(question, padding=question_padding).last_layer

        context_encoded = self.bilstm_before_coattention(context_embedded, context_lengths)[0]
        question_encoded = self.bilstm_before_coattention(question_embedded, question_lengths)[0]

        coattended_context, coattended_question = self.coattention(context_encoded, question_encoded,
                                                                   context_padding, question_padding)

        context_summary = torch.cat([coattended_context, context_encoded, context_embedded], -1)
        condensed_context, _ = self.context_bilstm_after_coattention(context_summary, context_lengths)
        self_attended_context = self.self_attentive_encoder_context(condensed_context, padding=context_padding)
        final_context, (context_rnn_h, context_rnn_c) = self.bilstm_context(self_attended_context[-1],
                                                                            context_lengths)
        context_rnn_state = [self.reshape_rnn_state(x) for x in (context_rnn_h, context_rnn_c)]

        question_summary = torch.cat([coattended_question, question_encoded, question_embedded], -1)
        condensed_question, _ = self.question_bilstm_after_coattention(question_summary, question_lengths)
        self_attended_question = self.self_attentive_encoder_question(condensed_question, padding=question_padding)
        final_question, (question_rnn_h, question_rnn_c) = self.bilstm_question(self_attended_question[-1],
                                                                                question_lengths)
        question_rnn_state = [self.reshape_rnn_state(x) for x in (question_rnn_h, question_rnn_c)]

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
