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

import math
import torch
from torch import nn
from torch.nn import functional as F

from .common import CombinedEmbedding, TransformerDecoder, LSTMDecoderAttention, Feedforward, \
    mask, positional_encodings_like, EPSILON, MultiLSTMCell


class MQANDecoder(nn.Module):
    def __init__(self, numericalizer, args, decoder_embeddings):
        super().__init__()
        self.numericalizer = numericalizer
        self.pad_idx = numericalizer.pad_id
        self.init_idx = numericalizer.init_id
        self.args = args

        self.decoder_embeddings = CombinedEmbedding(numericalizer, decoder_embeddings, args.dimension,
                                                    trained_dimension=args.trainable_decoder_embeddings,
                                                    project=True,
                                                    finetune_pretrained=False)

        if args.transformer_layers > 0:
            self.self_attentive_decoder = TransformerDecoder(args.dimension, args.transformer_heads,
                                                             args.transformer_hidden,
                                                             args.transformer_layers,
                                                             args.dropout_ratio)
        else:
            self.self_attentive_decoder = None

        if args.rnn_layers > 0:
            self.rnn_decoder = LSTMDecoder(args.dimension, args.rnn_dimension,
                                           dropout=args.dropout_ratio, num_layers=args.rnn_layers)
            switch_input_len = 2 * args.rnn_dimension + args.dimension
        else:
            self.context_attn = LSTMDecoderAttention(args.dimension, dot=True)
            self.question_attn = LSTMDecoderAttention(args.dimension, dot=True)
            self.dropout = nn.Dropout(args.dropout_ratio)
            switch_input_len = 2 * args.dimension
        self.vocab_pointer_switch = nn.Sequential(Feedforward(switch_input_len, 1), nn.Sigmoid())
        self.context_question_switch = nn.Sequential(Feedforward(switch_input_len, 1), nn.Sigmoid())

        self.generative_vocab_size = numericalizer.generative_vocab_size
        self.out = nn.Linear(args.rnn_dimension if args.rnn_layers > 0 else args.dimension, self.generative_vocab_size)

    def set_embeddings(self, embeddings):
        if self.decoder_embeddings is not None:
            self.decoder_embeddings.set_embeddings(embeddings)

    def forward(self, batch, self_attended_context, final_context, context_rnn_state, final_question,
                question_rnn_state):
        context, context_lengths, context_limited = batch.context.value, batch.context.length, batch.context.limited
        question, question_lengths, question_limited = batch.question.value, batch.question.length, batch.question.limited
        answer, answer_lengths, answer_limited = batch.answer.value, batch.answer.length, batch.answer.limited
        decoder_vocab = batch.decoder_vocab

        self.map_to_full = decoder_vocab.decode

        context_padding = context.data == self.pad_idx
        question_padding = question.data == self.pad_idx

        if self.args.rnn_layers > 0:
            self.rnn_decoder.applyMasks(context_padding, question_padding)
        else:
            self.context_attn.applyMasks(context_padding)
            self.question_attn.applyMasks(question_padding)

        if self.training:
            answer_padding = (answer.data == self.pad_idx)[:, :-1]

            answer_embedded = self.decoder_embeddings(answer[:, :-1], padding=answer_padding).last_layer

            if self.args.transformer_layers > 0:
                self_attended_decoded = self.self_attentive_decoder(answer_embedded,
                                                                    self_attended_context,
                                                                    context_padding=context_padding,
                                                                    answer_padding=answer_padding,
                                                                    positional_encodings=True)
            else:
                self_attended_decoded = answer_embedded

            if self.args.rnn_layers > 0:
                rnn_decoder_outputs = self.rnn_decoder(self_attended_decoded, final_context, final_question,
                                                       hidden=context_rnn_state)
                decoder_output, vocab_pointer_switch_input, context_question_switch_input, context_attention, \
                question_attention, rnn_state = rnn_decoder_outputs
            else:
                context_decoder_output, context_attention = self.context_attn(self_attended_decoded, final_context)
                question_decoder_output, question_attention = self.question_attn(self_attended_decoded, final_question)

                vocab_pointer_switch_input = torch.cat((context_decoder_output, self_attended_decoded), dim=-1)
                context_question_switch_input = torch.cat((question_decoder_output, self_attended_decoded), dim=-1)

                decoder_output = self.dropout(context_decoder_output)

            vocab_pointer_switch = self.vocab_pointer_switch(vocab_pointer_switch_input)
            context_question_switch = self.context_question_switch(context_question_switch_input)

            probs = self.probs(decoder_output, vocab_pointer_switch, context_question_switch,
                               context_attention, question_attention,
                               context_limited, question_limited,
                               decoder_vocab)

            probs, targets = mask(answer_limited[:, 1:].contiguous(), probs.contiguous(), pad_idx=decoder_vocab.pad_idx)
            loss = F.nll_loss(probs.log(), targets)
            return loss, None

        else:
            return None, self.greedy(self_attended_context, final_context, context_padding, final_question,
                                     context_limited, question_limited,
                                     decoder_vocab, rnn_state=context_rnn_state).data

    def probs(self, outputs, vocab_pointer_switches, context_question_switches,
              context_attention, question_attention,
              context_indices, question_indices,
              decoder_vocab):

        size = list(outputs.size())

        size[-1] = self.generative_vocab_size
        scores = self.out(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim() - 1)
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = len(decoder_vocab)
        if self.generative_vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = scaled_p_vocab.new_full(size, EPSILON)
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim() - 1)

        # p_context_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim() - 1, context_indices.unsqueeze(1).expand_as(context_attention),
                                    (context_question_switches * (1 - vocab_pointer_switches)).expand_as(
                                        context_attention) * context_attention)

        # p_question_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim() - 1,
                                    question_indices.unsqueeze(1).expand_as(question_attention),
                                    ((1 - context_question_switches) * (1 - vocab_pointer_switches)).expand_as(
                                        question_attention) * question_attention)

        return scaled_p_vocab

    def greedy(self, self_attended_context, context, context_padding, question, context_indices, question_indices,
               decoder_vocab, rnn_state=None):
        batch_size = context.size()[0]
        max_decoder_time = self.args.max_output_length
        outs = context.new_full((batch_size, max_decoder_time), self.pad_idx, dtype=torch.long)

        if self.args.transformer_layers > 0:
            hiddens = [self_attended_context[0].new_zeros((batch_size, max_decoder_time, self.args.dimension))
                       for l in range(len(self.self_attentive_decoder.layers) + 1)]
            hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        eos_yet = context.new_zeros((batch_size,)).byte()

        decoder_output = None
        for t in range(max_decoder_time):
            if t == 0:
                init_token = self_attended_context[-1].new_full((batch_size, 1), self.init_idx, dtype=torch.long)
                embedding = self.decoder_embeddings(init_token).last_layer
            else:
                current_token_id = outs[:, t - 1].unsqueeze(1)
                embedding = self.decoder_embeddings(current_token_id).last_layer

            if self.args.transformer_layers > 0:
                hiddens[0][:, t] = hiddens[0][:, t] + \
                                   (math.sqrt(self.self_attentive_decoder.d_model) * embedding).squeeze(1)
                for l in range(len(self.self_attentive_decoder.layers)):
                    hiddens[l + 1][:, t] = self.self_attentive_decoder.layers[l](hiddens[l][:, t],
                                                                                 self_attended_context[l],
                                                                                 selfattn_keys=hiddens[l][:, :t + 1],
                                                                                 context_padding=context_padding)

                self_attended_decoded = hiddens[-1][:, t].unsqueeze(1)
            else:
                self_attended_decoded = embedding

            if self.args.rnn_layers > 0:
                rnn_decoder_outputs = self.rnn_decoder(self_attended_decoded, context, question,
                                                       hidden=rnn_state, output=decoder_output)
                decoder_output, vocab_pointer_switch_input, context_question_switch_input, context_attention, \
                question_attention, rnn_state = rnn_decoder_outputs
            else:
                context_decoder_output, context_attention = self.context_attn(self_attended_decoded, context)
                question_decoder_output, question_attention = self.question_attn(self_attended_decoded, question)

                vocab_pointer_switch_input = torch.cat((context_decoder_output, self_attended_decoded), dim=-1)
                context_question_switch_input = torch.cat((question_decoder_output, self_attended_decoded), dim=-1)

                decoder_output = self.dropout(context_decoder_output)

            vocab_pointer_switch = self.vocab_pointer_switch(vocab_pointer_switch_input)
            context_question_switch = self.context_question_switch(context_question_switch_input)

            probs = self.probs(decoder_output, vocab_pointer_switch, context_question_switch,
                               context_attention, question_attention,
                               context_indices, question_indices, decoder_vocab)
            pred_probs, preds = probs.max(-1)
            preds = preds.squeeze(1)
            eos_yet = eos_yet | (preds == decoder_vocab.eos_idx).byte()
            outs[:, t] = preds.cpu().apply_(self.map_to_full)
            if eos_yet.all():
                break
        return outs


class LSTMDecoder(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.0, num_layers=1):
        super().__init__()
        self.d_hid = d_hid
        self.d_in = d_in
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.input_feed = True
        if self.input_feed:
            d_in += 1 * d_hid

        self.rnn = MultiLSTMCell(self.num_layers, d_in, d_hid, dropout)
        self.context_attn = LSTMDecoderAttention(d_hid, dot=True)
        self.question_attn = LSTMDecoderAttention(d_hid, dot=True)

    def applyMasks(self, context_mask, question_mask):
        self.context_attn.applyMasks(context_mask)
        self.question_attn.applyMasks(question_mask)

    def forward(self, input: torch.Tensor, context, question, output=None, hidden=None):
        context_output = output if output is not None else self.make_init_output(context)

        context_outputs, vocab_pointer_switch_inputs, context_question_switch_inputs, context_attentions, question_attentions = [], [], [], [], []
        for decoder_input in input.split(1, dim=1):
            context_output = self.dropout(context_output)
            if self.input_feed:
                rnn_input = torch.cat([decoder_input, context_output], 2)
            else:
                rnn_input = decoder_input

            rnn_input = rnn_input.squeeze(1)
            dec_state, hidden = self.rnn(rnn_input, hidden)
            dec_state = dec_state.unsqueeze(1)

            context_output, context_attention = self.context_attn(dec_state, context)
            question_output, question_attention = self.question_attn(dec_state, question)
            vocab_pointer_switch_inputs.append(torch.cat([dec_state, context_output, decoder_input], -1))
            context_question_switch_inputs.append(torch.cat([dec_state, question_output, decoder_input], -1))

            context_output = self.dropout(context_output)
            context_outputs.append(context_output)
            context_attentions.append(context_attention)
            question_attentions.append(question_attention)

        return [torch.cat(x, dim=1) for x in (context_outputs,
                                              vocab_pointer_switch_inputs,
                                              context_question_switch_inputs,
                                              context_attentions,
                                              question_attentions)] + [hidden]

    def make_init_output(self, context):
        batch_size = context.size(0)
        h_size = (batch_size, 1, self.d_hid)
        return context.new_zeros(h_size)
