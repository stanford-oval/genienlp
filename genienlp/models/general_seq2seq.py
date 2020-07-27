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
        question_embeddings = kwargs.pop("question_embeddings", None)
        decoder_embeddings = kwargs.pop("decoder_embeddings", None)
        save_dict = torch.load(args.best_checkpoint, map_location=device)
        model = Seq2Seq(numericalizer, args, context_embeddings, question_embeddings, decoder_embeddings)
        model.load_state_dict(save_dict['model_state_dict'])
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

        context_logits = context_logits.view(-1, self.numericalizer.num_tokens)
        masked_labels = masked_labels.view(-1)
        loss = torch.nn.functional.cross_entropy(context_logits, masked_labels, ignore_index=self.numericalizer.pad_id)
        return (loss, )

    def _normal_forward(self, batch, current_token_id, past=None, expansion_factor=1, generation_dict=None, encoder_output=None):
        if encoder_output is None:
            self_attended_context, final_context, context_rnn_state, final_question, question_rnn_state = self.encoder(batch)
        else:
            self_attended_context, final_context, context_rnn_state, final_question, question_rnn_state = encoder_output
        encoder_loss = None
        if self.training and getattr(self.args, 'use_encoder_loss', None):
            encoder_loss = self.get_encoder_loss(context_rnn_state)
        return self.decoder(batch, self_attended_context, final_context, context_rnn_state,
                            final_question, question_rnn_state, encoder_loss, current_token_id, decoder_wrapper=past,
                            expansion_factor=expansion_factor, generation_dict=generation_dict)

    def forward(self, batch, pretraining=False, current_token_id=None, past=None,
                expansion_factor=1, generation_dict=None, encoder_output=None):
        """
        When training or pretraining, forward() always returns a tuple where the first element is `loss`
        """
        if pretraining:
            return self._pretrain_forward(batch)
        else:
            return self._normal_forward(batch, current_token_id, past, expansion_factor, generation_dict, encoder_output)
        
        
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

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, batch, generation_dict, encoder_output):
        expansion_factor = input_ids.shape[0] // len(batch.example_id)
        return {"batch": batch, "past": past, "current_token_id": input_ids[:,-1:], "expansion_factor": expansion_factor, "generation_dict": generation_dict, "encoder_output": encoder_output}

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
        self.config.vocab_size = len(batch.decoder_vocab)
        self.config.is_encoder_decoder = False # in order to make it work with `transformers` generation code, we should treat this as a decoder-only model
        batch_size = len(batch.example_id)
        input_ids = torch.full((batch_size, 1), self.decoder.init_idx, dtype=torch.long, device=batch.context.value.device)
        
        generated = super().generate(input_ids=input_ids,
                                     batch=batch,
                                     max_length=max_output_length,
                                     min_length=2, # generate at least one token after BOS
                                     bos_token_id=self.decoder.init_idx,
                                     pad_token_id=batch.decoder_vocab.pad_idx,
                                     early_stopping=True,
                                     num_return_sequences=num_outputs,
                                     repetition_penalty=repetition_penalty,
                                     temperature=temperature,
                                     eos_token_id=batch.decoder_vocab.eos_idx,
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
        

    
            
        
    
    
    
    
    
    
    
    
