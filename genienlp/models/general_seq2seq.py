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
from .mqan_decoder import MQANDecoder, MQANDecoderWrapper
from .common import mask_tokens
from transformers import PreTrainedModel, PretrainedConfig

ENCODERS = {
    'MQANEncoder': MQANEncoder,
    'BiLSTM': BiLSTMEncoder,
    'Identity': IdentityEncoder,
    'Coattention': CoattentionEncoder,
}
DECODERS = {
    'MQANDecoder': MQANDecoder
}


class Seq2Seq(PreTrainedModel):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        args = kwargs.pop("args", None)
        device = kwargs.pop("device", None)
        numericalizer = kwargs.pop("numericalizer", None)
        context_embeddings = kwargs.pop("context_embeddings", None)
        question_embeddings = kwargs.pop("question_embedding", None)
        decoder_embeddings = kwargs.pop("decoder_embeddings", None)
        # print('args = ', args)
        # print('pretrained_model_name_or_path = ', pretrained_model_name_or_path)
        save_dict = torch.load(args.best_checkpoint, map_location=device)
        # print(save_dict)
        model = Seq2Seq(numericalizer, args, context_embeddings, question_embeddings, decoder_embeddings)
        model_dict = save_dict['model_state_dict']
        model.load_state_dict(model_dict)
        return model

    def __init__(self, numericalizer, args, context_embeddings, question_embeddings, decoder_embeddings):
        super().__init__(PretrainedConfig()) # dummy PretrainedConfig
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

    def _normal_forward(self, batch, current_token_id, past=None):
        self_attended_context, final_context, context_rnn_state, final_question, question_rnn_state = \
            self.encoder(batch)
        encoder_loss = None
        if getattr(self.args, 'use_encoder_loss', None) and self.training:
            encoder_loss = self.get_encoder_loss(context_rnn_state)
        return self.decoder(batch, self_attended_context, final_context, context_rnn_state,
                            final_question, question_rnn_state, encoder_loss, current_token_id, decoder_wrapper=past)

    # TODO iteration is unused, remove it
    def forward(self, batch, iteration=0, pretraining=False, current_token_id=None, past=None):
        # print('batch = ', batch)
        if pretraining:
            return self._pretrain_forward(batch)
        else:
            return self._normal_forward(batch, current_token_id, past)
        
        
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

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, batch):
        # print('input_ids = ', input_ids)
        # exit(0)
        return {"batch": batch, "past": past, "current_token_id": input_ids[:,-1:]}

    def _reorder_cache(self, past, beam_idx):
        past.reorder(beam_idx)
        return past

    def generate(self, batch, **kwargs):
        # print('batch = ', batch)
        self.config.vocab_size = len(batch.decoder_vocab)
        batch_size = len(batch.example_id)
        input_ids = torch.full((batch_size, 1), self.decoder.init_idx, dtype=torch.long, device=batch.context.value.device)

        # print('self.args.num_beams = ', self.args.num_beams)
        generated = super().generate(input_ids=input_ids, bos_token_id=self.decoder.init_idx, batch=batch, do_sample=False, temperature=1,
                                    eos_token_id=batch.decoder_vocab.eos_idx, num_beams=self.args.num_beams, max_length=self.args.max_output_length,
                                    top_k=0, top_p=1, pad_token_id=batch.decoder_vocab.pad_idx)
        generated = generated[:, 1:].cpu().apply_(self.decoder.map_to_full).to(batch.context.value.device) # remove bos and map to full vocabulary
        return generated
        

    
            
        
    
    
    
    
    
    
    
    
