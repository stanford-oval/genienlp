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

from .coatt_encoder import CoattentionEncoder
from .lstm_encoder import BiLSTMEncoder
from .mqan_encoder import MQANEncoder
from .identity_encoder import IdentityEncoder
from .mqan_decoder import MQANDecoder
from .common import mask_tokens

ENCODERS = {
    'MQANEncoder': MQANEncoder,
    'BiLSTM': BiLSTMEncoder,
    'Identity': IdentityEncoder,
    'Coattention': CoattentionEncoder,
}
DECODERS = {
    'MQANDecoder': MQANDecoder
}


class Seq2Seq(torch.nn.Module):
    def __init__(self, numericalizer, args, context_embeddings, question_embeddings, decoder_embeddings):
        super().__init__()
        self.args = args

        self.numericalizer = numericalizer
        self.encoder = ENCODERS[args.seq2seq_encoder](numericalizer, args, context_embeddings, question_embeddings)
        self.decoder = DECODERS[args.seq2seq_decoder](numericalizer, args, decoder_embeddings)

        if self.args.pretrain_context > 0:
            self.context_pretrain_lm_head = torch.nn.Linear(self.args.dimension, numericalizer.num_tokens)

    def set_train_context_embeddings(self, trainable):
        self.encoder.set_train_context_embeddings(trainable)

    def set_train_question_embeddings(self, trainable):
        self.encoder.set_train_question_embeddings(trainable)

    def _pretrain_forward(self, batch):
        masked_input, masked_labels = mask_tokens(batch.context.value, self.numericalizer,
                                                  self.args.pretrain_mlm_probability)
        masked_batch = batch._replace(context=batch.context._replace(value=masked_input))

        self_attended_context, _final_context, _context_rnn_state, _final_question, _question_rnn_state = \
            self.encoder(masked_batch)
        context_logits = self.context_pretrain_lm_head(self_attended_context[-1])
        predictions = None

        context_logits = context_logits.view(-1, self.numericalizer.num_tokens)
        masked_labels = masked_labels.view(-1)
        loss = torch.nn.functional.cross_entropy(context_logits, masked_labels, ignore_index=self.numericalizer.pad_id)
        return loss, predictions

    def _normal_forward(self, batch):
        self_attended_context, final_context, context_rnn_state, final_question, question_rnn_state = \
            self.encoder(batch)
        return self.decoder(batch, self_attended_context, final_context, context_rnn_state,
                            final_question, question_rnn_state)

    def forward(self, batch, iteration, pretraining=False):
        if pretraining:
            return self._pretrain_forward(batch)
        else:
            return self._normal_forward(batch)
