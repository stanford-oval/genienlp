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
from typing import NamedTuple, List

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.jit import Final
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class EmbeddingOutput(NamedTuple):
    all_layers: List[torch.Tensor]
    last_layer: torch.Tensor


INF = 1e10
EPSILON = 1e-10


class MultiLSTMCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(MultiLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            input = self.dropout(input)
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0., x.size(1))
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)


# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)


def pad_to_match(x, y):
    x_len, y_len = x.size(1), y.size(1)
    if x_len == y_len:
        return x, y
    extra = x.new_ones((x.size(0), abs(y_len - x_len)))
    if x_len < y_len:
        return torch.cat((x, extra), 1), y
    return x, torch.cat((y, extra), 1)


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, dropout_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x, padding=None):
        return self.layernorm(x[0] + self.dropout(self.layer(*x, padding=padding)))


class Attention(nn.Module):

    def __init__(self, d_key, dropout_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(dropout_ratio)
        self.causal = causal

    def forward(self, query, key, value, padding=None):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and self.causal:
            tri = key.new_ones((key.size(1), key.size(1))).triu(1) * INF
            dot_products.sub_(tri.unsqueeze(0))
        if padding is not None:
            if dot_products.dim() == 3:
                # dot_products is batch x query time x key time
                # padding is batch x key time
                # unsqueeze to batch x 1 x key time then broadcast
                dot_products.masked_fill_(padding.unsqueeze(1).expand_as(dot_products), -INF)
            else:
                # dot_products is batch x key time and is directly compatible with padding
                dot_products.masked_fill_(padding, -INF)
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)


class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, dropout_ratio, causal=False):
        super().__init__()
        self.attention = Attention(d_key, dropout_ratio, causal=causal)
        self.wq = Linear(d_key, d_key, bias=False)
        self.wk = Linear(d_key, d_key, bias=False)
        self.wv = Linear(d_value, d_value, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value, padding=None):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return torch.cat([self.attention(q, k, v, padding=padding)
                          for q, k, v in zip(query, key, value)], -1)


class LinearReLU(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.feedforward = Feedforward(d_model, d_hidden, activation='relu')
        self.linear = Linear(d_hidden, d_model)

    def forward(self, x, padding=None):
        return self.linear(self.feedforward(x))


class TransformerEncoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(
                dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, x, padding=None):
        return self.feedforward(self.selfattn(x, x, x, padding=padding))


class TransformerEncoder(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dimension, n_heads, hidden, dropout) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding=None):
        x = self.dropout(x)
        encoding = [x]
        for layer in self.layers:
            x = layer(x, padding=padding)
            encoding.append(x)
        return encoding


class TransformerDecoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout, causal=True):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout, causal),
            dimension, dropout)
        self.attention = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, x, encoding, selfattn_keys=None, context_padding=None, answer_padding=None):
        if selfattn_keys is None:
            selfattn_keys = x
        x = self.selfattn(x, selfattn_keys, selfattn_keys, padding=answer_padding)
        return self.feedforward(self.attention(x, encoding, encoding, padding=context_padding))


class TransformerDecoder(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout, causal=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(dimension, n_heads, hidden, dropout, causal=causal) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = dimension

    def forward(self, x, encoding, context_padding=None, positional_encodings=True, answer_padding=None):
        if positional_encodings:
            x = x + positional_encodings_like(x)
        x = self.dropout(x)
        for layer, enc in zip(self.layers, encoding[1:]):
            x = layer(x, enc, context_padding=context_padding, answer_padding=answer_padding)
        return x


def mask(targets, out, squash=True, pad_idx=1):
    mask = (targets != pad_idx)
    out_mask = mask.unsqueeze(-1).expand_as(out).contiguous()
    if squash:
        out_after = out[out_mask].contiguous().view(-1, out.size(-1))
    else:
        out_after = out * out_mask.float()
    targets_after = targets[mask]
    return out_after, targets_after


class Highway(torch.nn.Module):
    def __init__(self, d_in, activation='relu', n_layers=1):
        super(Highway, self).__init__()
        self.d_in = d_in
        self._layers = torch.nn.ModuleList([Linear(d_in, 2 * d_in) for _ in range(n_layers)])
        for layer in self._layers:
            layer.bias[d_in:].fill_(1)
        self.activation = getattr(F, activation)

    def forward(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, :self.d_in] if projected_input.dim() == 2 \
                else projected_input[:, :, :self.d_in]
            nonlinear_part = self.activation(nonlinear_part)
            gate = projected_input[:, self.d_in:(2 * self.d_in)] if projected_input.dim() == 2 \
                else projected_input[:, :, self.d_in:(2 * self.d_in)]
            gate = F.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class LinearFeedforward(nn.Module):

    def __init__(self, d_in, d_hid, d_out, activation='relu', dropout=0.2):
        super().__init__()
        self.feedforward = Feedforward(d_in, d_hid, activation=activation)
        self.linear = Linear(d_hid, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.feedforward(x)))


class PackedLSTM(nn.Module):

    def __init__(self, d_in, d_out, bidirectional=False, num_layers=1,
                 dropout=0.0, batch_first=True):
        """A wrapper class that packs input sequences and unpacks output sequences"""
        super().__init__()
        if bidirectional:
            d_out = d_out // 2
        self.rnn = nn.LSTM(d_in, d_out,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.batch_first = batch_first

    def forward(self, inputs, lengths, hidden=None):
        lens, indices = torch.sort(lengths.clone().detach(), 0, True)
        inputs = inputs[indices] if self.batch_first else inputs[:, indices]
        outputs, (h, c) = self.rnn(pack(inputs, lens.tolist(),
                                        batch_first=self.batch_first), hidden)
        outputs = unpack(outputs, batch_first=self.batch_first)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:, _indices]
        h, c = h[:, _indices, :], c[:, _indices, :]
        return outputs, (h, c)


class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


class Feedforward(nn.Module):

    def __init__(self, d_in, d_out, activation=None, bias=True, dropout=0.2):
        super().__init__()
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = lambda x: x
        self.linear = Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.activation(self.linear(self.dropout(x)))


class CombinedEmbedding(nn.Module):
    project: Final[bool]
    dimension: Final[int]

    def __init__(self, numericalizer, pretrained_embeddings,
                 output_dimension,
                 finetune_pretrained=False,
                 trained_dimension=0,
                 project=True):
        super().__init__()
        self.project = project
        self.pretrained_embeddings = nn.ModuleList(pretrained_embeddings)

        dimension = 0
        for idx, embedding in enumerate(self.pretrained_embeddings):
            dimension += embedding.dim
        self.set_trainable(finetune_pretrained)

        if trained_dimension > 0:
            self.trained_embeddings = nn.Embedding(numericalizer.num_tokens, trained_dimension)
            dimension += trained_dimension
        else:
            self.trained_embeddings = None
        if self.project:
            self.projection = Feedforward(dimension, output_dimension)
        else:
            assert dimension == output_dimension, (dimension, output_dimension)
        self.dimension = output_dimension

    def set_trainable(self, trainable):
        self.pretrained_embeddings.requires_grad_(trainable)

    def _combine_embeddings(self, embeddings):
        if len(embeddings) == 1:
            all_layers = embeddings[0].all_layers
            last_layer = embeddings[0].last_layer
            if self.project:
                last_layer = self.projection(last_layer)
            return EmbeddingOutput(all_layers=all_layers, last_layer=last_layer)

        all_layers = None
        last_layer = []
        for emb in embeddings:
            if all_layers is None:
                all_layers = [[layer] for layer in emb.all_layers]
            elif len(all_layers) != len(emb.all_layers):
                raise ValueError('Cannot combine embeddings that use different numbers of layers')
            else:
                for layer_list, layer in zip(all_layers, emb.all_layers):
                    layer_list.append(layer)
            last_layer.append(emb.last_layer)

        all_layers = [torch.cat(layer, dim=2) for layer in all_layers]
        last_layer = torch.cat(last_layer, dim=2)
        if self.project:
            last_layer = self.projection(last_layer)
        return EmbeddingOutput(all_layers=all_layers, last_layer=last_layer)

    def forward(self, x, padding=None):
        embedded: List[EmbeddingOutput] = []
        if self.pretrained_embeddings is not None:
            embedded += [emb(x, padding=padding) for emb in self.pretrained_embeddings]

        if self.trained_embeddings is not None:
            trained_vocabulary_size = self.trained_embeddings.weight.size()[0]
            valid_x = torch.lt(x, trained_vocabulary_size)
            masked_x = torch.where(valid_x, x, torch.zeros_like(x))
            output = self.trained_embeddings(masked_x)
            embedded.append(EmbeddingOutput(all_layers=[output], last_layer=output))

        return self._combine_embeddings(embedded)


class SemanticFusionUnit(nn.Module):

    def __init__(self, d, l):
        super().__init__()
        self.r_hat = Feedforward(d * l, d, 'tanh')
        self.g = Feedforward(d * l, d, 'sigmoid')
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        c = self.dropout(torch.cat(x, -1))
        r_hat = self.r_hat(c)
        g = self.g(c)
        o = g * r_hat + (1 - g) * x[0]
        return o


class LSTMDecoderAttention(nn.Module):
    def __init__(self, dim, dot=False):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(2 * dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.dot = dot

    def applyMasks(self, context_mask):
        # context_mask is batch x encoder_time, convert it to batch x 1 x encoder_time
        self.context_mask = context_mask.unsqueeze(1)

    def forward(self, input: torch.Tensor, context: torch.Tensor):
        # input is batch x decoder_time x dim
        # context is batch x encoder_time x dim
        # output will be batch x decoder_time x dim
        # context_attention will be batch x decoder_time x encoder_time

        if not self.dot:
            targetT = self.linear_in(input)  # batch x decoder_time x dim x 1
        else:
            targetT = input

        transposed_context = torch.transpose(context, 2, 1)
        context_scores = torch.matmul(targetT, transposed_context)
        context_scores.masked_fill_(self.context_mask, -float('inf'))
        context_attention = F.softmax(context_scores, dim=-1) + EPSILON

        # convert context_attention to batch x decoder_time x 1 x encoder_time
        # convert context to batch x 1 x encoder_time x dim
        # context_alignment will be batch x decoder_time x 1 x dim
        context_alignment = torch.matmul(context_attention.unsqueeze(2), context.unsqueeze(1))
        # squeeze out the extra dimension
        context_alignment = context_alignment.squeeze(2)

        combined_representation = torch.cat([input, context_alignment], 2)
        output = self.tanh(self.linear_out(combined_representation))

        return output, context_attention


class CoattentiveLayer(nn.Module):

    def __init__(self, d, dropout=0.2):
        super().__init__()
        self.proj = Feedforward(d, d, dropout=0.0)
        self.embed_sentinel = nn.Embedding(2, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, question, context_padding, question_padding):
        context_padding = torch.cat([context.new_zeros((context.size(0), 1), dtype=torch.bool), context_padding], 1)
        question_padding = torch.cat([question.new_zeros((question.size(0), 1), dtype=torch.bool), question_padding], 1)

        context_sentinel = self.embed_sentinel(context.new_zeros((context.size(0), 1), dtype=torch.long))
        context = torch.cat([context_sentinel, self.dropout(context)], 1) # batch_size x (context_length + 1) x features

        question_sentinel = self.embed_sentinel(question.new_ones((question.size(0), 1), dtype=torch.long))
        question = torch.cat([question_sentinel, question], 1)  # batch_size x (question_length + 1) x features
        question = torch.tanh(self.proj(question))  # batch_size x (question_length + 1) x features

        affinity = context.bmm(question.transpose(1, 2))  # batch_size x (context_length + 1) x (question_length + 1)
        attn_over_context = self.normalize(affinity, context_padding)  # batch_size x (context_length + 1) x 1
        attn_over_question = self.normalize(affinity.transpose(1, 2),
                                            question_padding)  # batch_size x (question_length + 1) x 1
        sum_of_context = self.attn(attn_over_context, context)  # batch_size x (question_length + 1) x features
        sum_of_question = self.attn(attn_over_question, question)  # batch_size x (context_length + 1) x features
        coattn_context = self.attn(attn_over_question, sum_of_context)  # batch_size x (context_length + 1) x features
        coattn_question = self.attn(attn_over_context, sum_of_question)  # batch_size x (question_length + 1) x features
        return torch.cat([coattn_context, sum_of_question], 2)[:, 1:], \
               torch.cat([coattn_question, sum_of_context], 2)[:, 1:]

    @staticmethod
    def attn(weights, candidates):
        w1, w2, w3 = weights.size()
        c1, c2, c3 = candidates.size()
        return weights.unsqueeze(3).expand(w1, w2, w3, c3).mul(candidates.unsqueeze(2).expand(c1, c2, w3, c3)) \
            .sum(1).squeeze(1)

    @staticmethod
    def normalize(original, padding):
        raw_scores = original.clone()
        raw_scores.masked_fill_(padding.unsqueeze(-1).expand_as(raw_scores), -INF)
        return F.softmax(raw_scores, dim=1)


# The following was copied and adapted from Hugginface's Tokenizers library
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def mask_tokens(inputs: torch.Tensor, numericalizer, probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    assert numericalizer.mask_id

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    probability_matrix = torch.full(labels.shape, probability, device=inputs.device)
    special_tokens_mask = [numericalizer.get_special_token_mask(token_ids) for token_ids in inputs.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=inputs.device),
                                    value=0.0)
    padding_mask = labels.eq(numericalizer.pad_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # All tokens that are not masked become padding, so we don't compute loss on them
    labels[~masked_indices] = numericalizer.pad_id

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool() & masked_indices
    inputs[indices_replaced] = numericalizer.mask_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(numericalizer.num_tokens, labels.shape, dtype=torch.long, device=inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        # print('hyp = ', hyp)
        # print('sum_logprobs = ', sum_logprobs)
        # print('len(hyp) ** self.length_penalty = ', len(hyp) ** self.length_penalty)
        score = sum_logprobs / len(hyp) ** self.length_penalty
        # print('score = ', score)
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[0][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
