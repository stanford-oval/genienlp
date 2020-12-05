#
# Copyright (c) 2018-2019, Salesforce, Inc.
#                          The Board of Trustees of the Leland Stanford Junior University
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
import logging
from transformers import AutoTokenizer, AutoModel, AutoConfig, \
    BERT_PRETRAINED_MODEL_ARCHIVE_LIST, XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from typing import NamedTuple, List

from .numericalizer.transformer import BertNumericalizer, XLMRobertaNumericalizer

logger = logging.getLogger(__name__)

EMBEDDING_NAME_TO_NUMERICALIZER_MAP = dict()
EMBEDDING_NAME_TO_NUMERICALIZER_MAP.update({embedding: BertNumericalizer for embedding in BERT_PRETRAINED_MODEL_ARCHIVE_LIST})
EMBEDDING_NAME_TO_NUMERICALIZER_MAP.update({embedding: XLMRobertaNumericalizer for embedding in XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST})


class EmbeddingOutput(NamedTuple):
    all_layers: List[torch.Tensor]
    last_layer: torch.Tensor


class TransformerEmbedding(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        model.config.output_hidden_states = True
        self.dim = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers
        self.model = model

    def init_for_vocab(self, vocab):
        self.model.resize_token_embeddings(len(vocab))

    def grow_for_vocab(self, vocab, new_words):
        self.model.resize_token_embeddings(len(vocab))

    def forward(self, input: torch.Tensor, padding=None):
        last_hidden_state, _pooled, hidden_states = self.model(input, attention_mask=(~padding).to(dtype=torch.float))

        return EmbeddingOutput(all_layers=hidden_states, last_layer=last_hidden_state)


def get_embedding_type(emb_name):
    if '@' in emb_name:
        return emb_name.split('@')[0]
    else:
        return emb_name
    

def load_embeddings(cachedir, context_emb_names, question_emb_names, decoder_emb_names,
                    max_generative_vocab=50000, cache_only=False):
    logger.info(f'Getting pretrained word vectors and pretrained models')

    context_emb_names = context_emb_names.split('+')
    question_emb_names = question_emb_names.split('+')
    decoder_emb_names = decoder_emb_names.split('+')

    all_vectors = {}
    context_vectors = []
    question_vectors = []
    decoder_vectors = []

    numericalizer = None
    numericalizer_type = None
    for emb_name in context_emb_names:
        if not emb_name:
            continue
        if emb_name in all_vectors:
            context_vectors.append(all_vectors[emb_name])
            continue

        emb_type = get_embedding_type(emb_name)
        if numericalizer is not None and numericalizer_type != emb_type and not cache_only:
            raise ValueError('Cannot specify multiple Transformer embeddings')

        config = AutoConfig.from_pretrained(emb_type, cache_dir=cachedir)
        config.output_hidden_states = True
        if numericalizer is None:
            numericalizer = EMBEDDING_NAME_TO_NUMERICALIZER_MAP[emb_type](emb_type, config=config,
                                              max_generative_vocab=max_generative_vocab,
                                              cache=cachedir)
            numericalizer_type = emb_type

        # load the tokenizer once to ensure all files are downloaded
        AutoTokenizer.from_pretrained(emb_type, cache_dir=cachedir)

        context_vectors.append(
            TransformerEmbedding(AutoModel.from_pretrained(emb_type, config=config, cache_dir=cachedir)))

    for emb_name in question_emb_names:
        if not emb_name:
            continue
        if emb_name in all_vectors:
            question_vectors.append(all_vectors[emb_name])
            continue

        emb_type = get_embedding_type(emb_name)
        if numericalizer is not None and numericalizer_type != emb_type:
            raise ValueError('Cannot specify multiple Transformer embeddings')

        config = AutoConfig.from_pretrained(emb_type, cache_dir=cachedir)
        config.output_hidden_states = True
        if numericalizer is None:
            numericalizer = EMBEDDING_NAME_TO_NUMERICALIZER_MAP[emb_type](emb_type, config=config,
                                              max_generative_vocab=max_generative_vocab,
                                              cache=cachedir)

            numericalizer_type = emb_type

        # load the tokenizer once to ensure all files are downloaded
        AutoTokenizer.from_pretrained(emb_type, cache_dir=cachedir)

        question_vectors.append(
            TransformerEmbedding(AutoModel.from_pretrained(emb_type, config=config, cache_dir=cachedir)))

    for emb_name in decoder_emb_names:
        if not emb_name:
            continue
        emb_type = get_embedding_type(emb_name)
        if emb_type in EMBEDDING_NAME_TO_NUMERICALIZER_MAP:
            raise ValueError('Transformer embeddings cannot be specified in the decoder')

    return numericalizer, context_vectors, question_vectors, decoder_vectors
