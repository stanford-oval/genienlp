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
from torch.nn import functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from .common import EPSILON, CombinedEmbedding, Feedforward, LSTMDecoderAttention, MultiLSTMCell, mask


class MQANDecoder(nn.Module):
    def __init__(self, numericalizer, args):
        super().__init__()
        self.numericalizer = numericalizer
        self.pad_idx = numericalizer.pad_id
        self.init_idx = numericalizer.init_id
        self.args = args

        self.decoder_embeddings = CombinedEmbedding(
            numericalizer, [], args.dimension, trained_dimension=args.trainable_decoder_embeddings, project=True
        )

        if args.rnn_layers > 0:
            self.rnn_decoder = LSTMDecoder(
                args.dimension, args.rnn_dimension, dropout=args.dropout_ratio, num_layers=args.rnn_layers
            )
            switch_input_len = 2 * args.rnn_dimension + args.dimension
        else:
            self.context_attn = LSTMDecoderAttention(args.dimension, dot=True)
            self.dropout = nn.Dropout(args.dropout_ratio)
            switch_input_len = 2 * args.dimension
        self.vocab_pointer_switch = nn.Sequential(Feedforward(switch_input_len, 1), nn.Sigmoid())

        self.generative_vocab_size = numericalizer.generative_vocab_size
        self.out = nn.Linear(args.rnn_dimension if args.rnn_layers > 0 else args.dimension, self.generative_vocab_size)

    def set_embeddings(self, embeddings):
        if self.decoder_embeddings is not None:
            self.decoder_embeddings.set_embeddings(embeddings)

    def forward(
        self,
        batch,
        final_context,
        context_rnn_state,
        encoder_loss,
        current_token_id=None,
        decoder_wrapper=None,
        expansion_factor=1,
        generation_dict=None,
    ):

        context, context_limited = batch.context.value, batch.context.limited
        answer, answer_limited = batch.answer.value, batch.answer.limited
        decoder_vocab = self.numericalizer.decoder_vocab
        self.map_to_full = decoder_vocab.decode
        context_padding = context.data == self.pad_idx
        if self.training:
            if self.args.rnn_layers > 0:
                self.rnn_decoder.applyMasks(context_padding)
            else:
                self.context_attn.applyMasks(context_padding)

            answer_padding = (answer.data == self.pad_idx)[:, :-1]
            answer_embedded = self.decoder_embeddings(answer[:, :-1], padding=answer_padding)

            if self.args.rnn_layers > 0:
                rnn_decoder_outputs = self.rnn_decoder(answer_embedded, final_context, hidden=context_rnn_state)
                decoder_output, vocab_pointer_switch_input, context_attention, rnn_state = rnn_decoder_outputs
            else:
                context_decoder_output, context_attention = self.context_attn(answer_embedded, final_context)
                vocab_pointer_switch_input = torch.cat((context_decoder_output, answer_embedded), dim=-1)
                decoder_output = self.dropout(context_decoder_output)

            vocab_pointer_switch = self.vocab_pointer_switch(vocab_pointer_switch_input)

            probs = self.probs(decoder_output, vocab_pointer_switch, context_attention, context_limited, decoder_vocab)

            probs, targets = mask(answer_limited[:, 1:].contiguous(), probs.contiguous(), pad_idx=decoder_vocab.pad_idx)
            loss = F.nll_loss(probs.log(), targets, ignore_index=decoder_vocab.pad_idx)
            if encoder_loss is not None:
                loss += self.args.encoder_loss_weight * encoder_loss

            return Seq2SeqLMOutput(loss=loss)
        else:
            if decoder_wrapper is None:
                decoder_wrapper = self.decoder_wrapper(
                    final_context,
                    context_padding,
                    context_limited,
                    decoder_vocab,
                    rnn_state=context_rnn_state,
                    expansion_factor=expansion_factor,
                    generation_dict=generation_dict,
                )
            else:
                current_token_id = current_token_id.clone().cpu().apply_(self.map_to_full).to(current_token_id.device)
            # (next_token_logits, past) where `past` includes all the states needed to continue generation
            logits = torch.log(decoder_wrapper.next_token_probs(current_token_id))
            return Seq2SeqLMOutput(logits=logits, past_key_values=decoder_wrapper)

    def probs(self, outputs, vocab_pointer_switches, context_attention, context_indices, decoder_vocab):
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
        scaled_p_vocab.scatter_add_(
            scaled_p_vocab.dim() - 1,
            context_indices.unsqueeze(1).expand_as(context_attention),
            ((1 - vocab_pointer_switches)).expand_as(context_attention) * context_attention,
        )

        return scaled_p_vocab

    def decoder_wrapper(
        self,
        context,
        context_padding,
        context_indices,
        decoder_vocab,
        rnn_state=None,
        expansion_factor=1,
        generation_dict=None,
    ):
        batch_size = context.size()[0]
        max_decoder_time = generation_dict['max_output_length']

        decoder_wrapper = MQANDecoderWrapper(
            context,
            context_padding,
            context_indices,
            decoder_vocab,
            rnn_state,
            batch_size,
            max_decoder_time,
            self,
            expansion_factor=expansion_factor,
        )

        return decoder_wrapper


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

    def applyMasks(self, context_mask):
        self.context_attn.applyMasks(context_mask)

    def forward(self, input: torch.Tensor, context, output=None, hidden=None):
        context_output = output if output is not None else self.make_init_output(context)

        context_outputs, vocab_pointer_switch_inputs, context_attentions = [], [], []
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
            vocab_pointer_switch_inputs.append(torch.cat([dec_state, context_output, decoder_input], -1))

            context_output = self.dropout(context_output)
            context_outputs.append(context_output)
            context_attentions.append(context_attention)

        return [torch.cat(x, dim=1) for x in (context_outputs, vocab_pointer_switch_inputs, context_attentions)] + [hidden]

    def make_init_output(self, context):
        batch_size = context.size(0)
        h_size = (batch_size, 1, self.d_hid)
        return context.new_zeros(h_size)


class MQANDecoderWrapper(object):
    """
    A wrapper for MQANDecoder that wraps around its recurrent neural network, so that we can decode it like a Transformer
    """

    def __init__(
        self,
        context,
        context_padding,
        context_indices,
        decoder_vocab,
        rnn_state,
        batch_size,
        max_decoder_time,
        mqan_decoder: MQANDecoder,
        expansion_factor: int,
    ):
        self.decoder_vocab = decoder_vocab
        context = self.expand_for_beam_search(context, batch_size, expansion_factor)
        context_padding = self.expand_for_beam_search(context_padding, batch_size, expansion_factor)
        context_indices = self.expand_for_beam_search(context_indices, batch_size, expansion_factor)
        if rnn_state is not None:
            rnn_state = self.expand_for_beam_search(rnn_state, batch_size, expansion_factor, dim=1)
        self.context = context
        self.context_padding = context_padding
        self.context_indices = context_indices
        self.rnn_state = rnn_state
        self.batch_size = batch_size
        self.max_decoder_time = max_decoder_time
        self.mqan_decoder = mqan_decoder

        if self.mqan_decoder.args.rnn_layers > 0:
            self.mqan_decoder.rnn_decoder.applyMasks(self.context_padding)
        else:
            self.mqan_decoder.context_attn.applyMasks(self.context_padding)

        self.time = 0
        self.decoder_output = None

    def reorder(self, new_order):
        # TODO only reordering rnn_state should be enough since reordering happens among beams of the same input
        self.context = self.reorder_for_beam_search(self.context, new_order)
        self.context_padding = self.reorder_for_beam_search(self.context_padding, new_order)
        self.context_indices = self.reorder_for_beam_search(self.context_indices, new_order)
        self.rnn_state = self.reorder_for_beam_search(self.rnn_state, new_order, dim=1)

        if self.mqan_decoder.args.rnn_layers > 0:
            self.mqan_decoder.rnn_decoder.applyMasks(self.context_padding)
        else:
            self.mqan_decoder.context_attn.applyMasks(self.context_padding)

    def next_token_probs(self, current_token_id):
        embedding = self.mqan_decoder.decoder_embeddings(current_token_id)

        if self.mqan_decoder.args.rnn_layers > 0:
            rnn_decoder_outputs = self.mqan_decoder.rnn_decoder(
                embedding, self.context, hidden=self.rnn_state, output=self.decoder_output
            )
            self.decoder_output, vocab_pointer_switch_input, context_attention, self.rnn_state = rnn_decoder_outputs
        else:
            context_decoder_output, context_attention = self.mqan_decoder.context_attn(embedding, self.context)
            vocab_pointer_switch_input = torch.cat((context_decoder_output, embedding), dim=-1)

            self.decoder_output = self.mqan_decoder.dropout(context_decoder_output)

        vocab_pointer_switch = self.mqan_decoder.vocab_pointer_switch(vocab_pointer_switch_input)

        probs = self.mqan_decoder.probs(
            self.decoder_output, vocab_pointer_switch, context_attention, self.context_indices, self.decoder_vocab
        )

        self.time += 1
        return probs

    def expand_for_beam_search(self, t, batch_size, num_beams, dim=0):
        if isinstance(t, tuple):
            elements = []
            for e in t:
                elements.append(self.expand_for_beam_search(e, batch_size, num_beams, dim))
            return tuple(elements)
        elif isinstance(t, list):
            elements = []
            for e in t:
                elements.append(self.expand_for_beam_search(e, batch_size, num_beams, dim))
            return elements

        original_size = list(t.shape)
        original_size[dim] *= num_beams
        t = t.unsqueeze(dim + 1)
        expanded_size = list(t.shape)
        expanded_size[dim + 1] = num_beams
        t = t.expand(*expanded_size)
        t = t.contiguous().view(*original_size)  # (batch_size * num_beams, -1)
        return t

    def reorder_for_beam_search(self, t, new_order, dim=0):
        if isinstance(t, tuple):
            elements = []
            for e in t:
                elements.append(self.reorder_for_beam_search(e, new_order, dim))
            return tuple(elements)
        elif isinstance(t, list):
            elements = []
            for e in t:
                elements.append(self.reorder_for_beam_search(e, new_order, dim))
            return elements

        p = [i for i in range(len(t.shape))]
        p[dim] = 0
        p[0] = dim
        t = t.permute(*p)
        t = t[new_order]
        t = t.permute(*p)

        return t
