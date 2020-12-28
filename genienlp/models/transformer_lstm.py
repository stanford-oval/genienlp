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
import logging
from transformers import AutoModel, PretrainedConfig, AutoConfig

from ..data_utils.numericalizer import TransformerNumericalizer
from .identity_encoder import IdentityEncoder
from .mqan_decoder import MQANDecoder
from .base import GenieModel

logger = logging.getLogger(__name__)


class TransformerLSTM(GenieModel):
    def __init__(self, config=None, *inputs, args, vocab_sets, tasks, save_directory=None, **kwargs):
        """
        Relevant inputs should be provided using kwargs. This method is defined this way to match parent's and siblings' method signatures.
        Inputs:
            args
            vocab_sets
            is_loading
            save_directory: The directory where numericalizer can be loaded from. Should be provided whenever `is_loading` is True
        """
        super().__init__(PretrainedConfig()) # dummy PretrainedConfig
        self.args = args

        encoder_embeddings = args.pretrained_model
        config = AutoConfig.from_pretrained(encoder_embeddings, cache_dir=args.embeddings)
        args.dimension = config.hidden_size
        self.numericalizer = TransformerNumericalizer(encoder_embeddings,
                                                      max_generative_vocab=args.max_generative_vocab,
                                                      cache=args.embeddings,
                                                      preprocess_special_tokens=args.preprocess_special_tokens)
        self.init_vocab_from_data(vocab_sets, tasks, save_directory)

        logger.info(f'Initializing encoder and decoder embeddings')
        if save_directory is not None:
            self.encoder_embeddings = AutoModel.from_config(config)
        else:
            self.encoder_embeddings = AutoModel.from_pretrained(encoder_embeddings, config=config, cache_dir=args.embeddings)
        self.encoder_embeddings.resize_token_embeddings(self.numericalizer.num_tokens)
        
        logger.info(f'Vocabulary has {self.numericalizer.num_tokens} tokens')

        self.encoder = IdentityEncoder(self.numericalizer, args, config, self.encoder_embeddings)
        self.decoder = MQANDecoder(self.numericalizer, args)

    def add_new_vocab_from_data(self, tasks, resize_decoder=False):
        super().add_new_vocab_from_data(tasks, resize_decoder=resize_decoder)
        self.encoder_embeddings.resize_token_embeddings(self.numericalizer.num_tokens)
        if resize_decoder:
            self.decoder.decoder_embeddings.resize_embedding(self.numericalizer.num_tokens)

    def forward(self, batch, current_token_id=None, past_key_values=None,
                expansion_factor=1, generation_dict=None, encoder_output=None, return_dict=False):
        if encoder_output is None:
            final_context, context_rnn_state = self.encoder(batch)
        else:
            final_context, context_rnn_state = encoder_output
        encoder_loss = None
        if self.training and getattr(self.args, 'use_encoder_loss', None):
            encoder_loss = self.get_encoder_loss(context_rnn_state)
            
        return self.decoder(batch, final_context, context_rnn_state,
                            encoder_loss, current_token_id, decoder_wrapper=past_key_values,
                            expansion_factor=expansion_factor, generation_dict=generation_dict)

    def get_encoder_loss(self, context_rnn_state):
        
        # concat hidden and cell state
        if len(context_rnn_state) == 2:
            context_rnn_state = torch.cat(context_rnn_state, dim=0)
            
        batch_size = context_rnn_state.size(1)
        groups = len(self.args.train_languages.split('+'))
        assert batch_size % groups == 0
        
        # reshape to be (batch_size; -1)
        context_rnn_state = context_rnn_state.view(batch_size, -1)
        
        if self.args.encoder_loss_type == 'mean':
            # element-wise mean of encoder loss https://www.aclweb.org/anthology/W18-3023.pdf
            context_value = torch.mean(context_rnn_state, dim=-1)
        elif self.args.encoder_loss_type == 'sum':
            context_value = torch.sum(context_rnn_state, dim=-1)
        
        encoder_loss = 0.0
        for i in range(0, batch_size, groups):
            indices = [j for j in range(i, i+groups)]
            groups_vals = context_value[indices]
            assert len(groups_vals) > 1
            encoder_loss += torch.std(groups_vals).item()
            
        return encoder_loss
        
    def get_output_embeddings(self):
        return self.decoder.decoder_embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask, use_cache, batch, generation_dict, encoder_output, past=None):
        expansion_factor = input_ids.shape[0] // len(batch.example_id)
        return {"batch": batch, "past_key_values": past, "current_token_id": input_ids[:,-1:], "expansion_factor": expansion_factor, "generation_dict": generation_dict, "encoder_output": encoder_output}

    def _reorder_cache(self, past, beam_idx):
        past.reorder(beam_idx)
        return past

    def generate(self,
                 batch,
                 max_output_length,
                 num_outputs,
                 temperature,
                 repetition_penalty,
                 top_k,
                 top_p,
                 num_beams,
                 no_repeat_ngram_size,
                 do_sample
                 ):

        encoder_output = self.encoder(batch)
        self.config.vocab_size = len(self.numericalizer.decoder_vocab)
        self.config.is_encoder_decoder = False # in order to make it work with `transformers` generation code, we should treat this as a decoder-only model
        batch_size = len(batch.example_id)
        input_ids = torch.full((batch_size, 1), self.decoder.init_idx, dtype=torch.long, device=batch.context.value.device)
        
        generated = super().generate(input_ids=input_ids,
                                     batch=batch,
                                     max_length=max_output_length,
                                     min_length=2, # generate at least one token after BOS
                                     bos_token_id=self.decoder.init_idx,
                                     pad_token_id=self.numericalizer.decoder_vocab.pad_idx,
                                     early_stopping=True,
                                     num_return_sequences=num_outputs,
                                     repetition_penalty=repetition_penalty,
                                     temperature=temperature,
                                     eos_token_id=self.numericalizer.decoder_vocab.eos_idx,
                                     top_k=top_k,
                                     top_p=top_p,
                                     num_beams=num_beams,
                                     no_repeat_ngram_size=no_repeat_ngram_size,
                                     do_sample=do_sample,
                                     generation_dict={'max_output_length': max_output_length, 'num_beams': num_beams},
                                     encoder_output=encoder_output
                                    )
        generated = torch.cat((generated[:, 0:1], generated[:, 1:].cpu().apply_(self.decoder.map_to_full).to(batch.context.value.device)), dim=1) # map everything to full vocabulary except BOS which already is in full vocabulary

        return generated
