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

from typing import List

import torch
import torch.nn as nn
from torch.jit import Final
from torch.nn import functional as F

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


# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)


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


def mask(targets, out, squash=True, pad_idx=1):
    mask = targets != pad_idx
    out_mask = mask.unsqueeze(-1).expand_as(out).contiguous()
    if squash:
        out_after = out[out_mask].contiguous().view(-1, out.size(-1))
    else:
        out_after = out * out_mask.float()
    targets_after = targets[mask]
    return out_after, targets_after


class LinearFeedforward(nn.Module):
    def __init__(self, d_in, d_hid, d_out, activation='relu', dropout=0.2):
        super().__init__()
        self.feedforward = Feedforward(d_in, d_hid, activation=activation)
        self.linear = Linear(d_hid, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.feedforward(x)))


class Linear(nn.Linear):
    def forward(self, x):
        size = x.size()
        return super().forward(x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


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

    def __init__(self, numericalizer, pretrained_embeddings, output_dimension, trained_dimension=0, project=True):
        super().__init__()
        self.project = project
        self.pretrained_embeddings = nn.ModuleList(pretrained_embeddings)

        dimension = 0
        for embedding in self.pretrained_embeddings:
            dimension += embedding.dim

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

    def resize_embedding(self, new_vocab_size):
        if self.trained_embeddings is None:
            # we are not training embeddings at all
            return
        dimensions = self.trained_embeddings.weight.shape
        if new_vocab_size == dimensions[0]:
            return
        assert new_vocab_size > dimensions[0], 'Cannot shrink the embedding matrix'
        resized_embeddings = nn.Embedding(new_vocab_size, dimensions[1])
        resized_embeddings.weight.data[0 : dimensions[0], :] = self.trained_embeddings.weight.data
        self.trained_embeddings = resized_embeddings

    def _combine_embeddings(self, embeddings):

        emb = torch.cat(embeddings, dim=2)
        if self.project:
            emb = self.projection(emb)
        return emb

    def forward(self, x, padding=None):
        embedded: List[torch.Tensor] = []
        if self.pretrained_embeddings is not None:
            embedded += [emb(x, padding=padding) for emb in self.pretrained_embeddings]

        if self.trained_embeddings is not None:
            trained_vocabulary_size = self.trained_embeddings.weight.size()[0]
            valid_x = torch.lt(x, trained_vocabulary_size)
            masked_x = torch.where(valid_x, x, torch.zeros_like(x))
            output = self.trained_embeddings(masked_x)
            embedded.append(output)

        return self._combine_embeddings(embedded)


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


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, smoothing):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target, ignore_index):
        """
        Inputs:
            x: Tensor of shape (N, vocab_size)
            target: Tensor of shape (N, ) where N is batch_size * sequence_length
            ignore_index: this index in the vocabulary is ignored when calculating loss. This is useful for pad tokens.
        Outputs:
            loss: a Tensor of shape (N, )
        """
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        loss.masked_fill_((target == ignore_index), 0)
        return loss
