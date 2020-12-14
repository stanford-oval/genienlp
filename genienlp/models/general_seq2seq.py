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
import os

from .coatt_encoder import CoattentionEncoder
from ..data_utils.embeddings import load_embeddings
from ..data_utils.numericalizer.transformer import BartNumericalizer, MBartNumericalizer, MT5Numericalizer
from ..data_utils.example import Example
from .lstm_encoder import BiLSTMEncoder
from .mqan_encoder import MQANEncoder
from .identity_encoder import IdentityEncoder
from .mqan_decoder import MQANDecoder
from .common import mask_tokens
from transformers import PreTrainedModel, PretrainedConfig, BartForConditionalGeneration, MBartForConditionalGeneration, \
    MT5ForConditionalGeneration

ENCODERS = {
    'MQANEncoder': MQANEncoder,
    'BiLSTM': BiLSTMEncoder,
    'Identity': IdentityEncoder,
    'Coattention': CoattentionEncoder,
}

DECODERS = {
    'MQANDecoder': MQANDecoder
}

logger = logging.getLogger(__name__)

class Seq2Seq(PreTrainedModel):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Inputs:
            pretrained_model_name_or_path: is the path to the directory where the model is saved.
                                           Is named this way to match the parent's and siblings' method signatures
            args
            device
            vocab_sets
        
        Outputs:
            model: the loaded model
            best_decascore: the best deca score when the training of this model was finished
        """

        # obtain function arguments from **kwargs
        save_directory = pretrained_model_name_or_path
        model_checkpoint_file = kwargs.pop("model_checkpoint_file", None)
        args = kwargs.pop("args", None)
        device = kwargs.pop("device", None)
        vocab_sets = kwargs.pop("vocab_sets", None)

        full_checkpoint_path = os.path.join(save_directory, model_checkpoint_file)
        logger.info(f'Loading the model from {full_checkpoint_path}')
        model = cls(args=args, vocab_sets=vocab_sets, is_loading=True, save_directory=save_directory)
        save_dict = torch.load(full_checkpoint_path, map_location=device)
        model.load_state_dict(save_dict['model_state_dict'])

        return model, save_dict.get('best_decascore')

    def __init__(self, config=None, *inputs, **kwargs):
        """
        Relevant inputs should be provided using kwargs. This method is defined this way to match parent's and siblings' method signatures.
        Inputs:
            args
            vocab_sets
            is_loading
            save_directory: The directory where numericalizer can be loaded from. Should be provided whenever `is_loading` is True
        """
        super().__init__(PretrainedConfig()) # dummy PretrainedConfig
        # obtain function arguments from **kwargs
        args = kwargs.pop("args", None)
        vocab_sets = kwargs.pop("vocab_sets", None)
        is_loading = kwargs.pop("is_loading", False)
        save_directory = kwargs.pop("save_directory", None)

        self.numericalizer, self.context_embeddings, self.question_embeddings, self.decoder_embeddings = \
            self._init_embeddings_from_data(args, vocab_sets, is_loading)
        self.args = args

        if is_loading:
            logger.info(f'Loading the accompanying numericalizer from {save_directory}')
            self.numericalizer.load(save_directory)

        logger.info(f'Initializing encoder and decoder embeddings')
        for vec in set(self.context_embeddings + self.question_embeddings + self.decoder_embeddings):
            vec.init_for_vocab(self.numericalizer.vocab)
        
        logger.info(f'Vocabulary has {self.numericalizer.num_tokens} tokens')
        logger.debug(f'The first 200 tokens:')
        logger.debug(self.numericalizer.vocab.itos[:200])

        self.encoder = ENCODERS[args.seq2seq_encoder](self.numericalizer, args, self.context_embeddings,
                                                      self.question_embeddings)
        self.decoder = DECODERS[args.seq2seq_decoder](self.numericalizer, args, self.decoder_embeddings)

        if self.args.pretrain_context > 0:
            self.context_pretrain_lm_head = torch.nn.Linear(self.args.dimension, self.numericalizer.num_tokens)

    def set_train_context_embeddings(self, trainable):
        self.encoder.set_train_context_embeddings(trainable)

    def add_new_vocab_from_data(self, splits, resize_decoder=False):
        """
        resize_decoder: if True, will actually resize the embedding matrix of the decoder
        """
        #logger.info(f'Vocabulary has {self.numericalizer.num_tokens} tokens from training')
        old_num_tokens = self.numericalizer.num_tokens
        new_words = []
        for task_splits in splits:
            for split in task_splits:
                new_words += self.numericalizer.grow_vocab(split)
        if self.numericalizer.num_tokens > old_num_tokens:
            logger.info(f'Vocabulary has expanded to {self.numericalizer.num_tokens} tokens')
        for emb in set(self.context_embeddings + self.question_embeddings + self.decoder_embeddings):
            emb.grow_for_vocab(self.numericalizer.vocab, new_words)
        if resize_decoder:
            self.decoder.decoder_embeddings.resize_embedding(self.numericalizer.num_tokens)

    def _init_embeddings_from_data(self, args, vocab_sets, is_loading):
        numericalizer, context_embeddings, question_embeddings, decoder_embeddings = \
        load_embeddings(args.embeddings,
                        args.context_embeddings,
                        args.question_embeddings,
                        args.decoder_embeddings,
                        args.max_generative_vocab)
        if not is_loading:
            logger.info(f'Building vocabulary')
            numericalizer.build_vocab(Example.vocab_fields, vocab_sets)
        else:
            pass # numericalizer is going to be loaded from a file, so no need to do anything

        return numericalizer, context_embeddings, question_embeddings, decoder_embeddings

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

    def _normal_forward(self, batch, current_token_id, past_key_values=None, expansion_factor=1, generation_dict=None,
                        encoder_output=None, return_dict=False):
        if encoder_output is None:
            self_attended_context, final_context, context_rnn_state, final_question, question_rnn_state = self.encoder(
                batch)
        else:
            self_attended_context, final_context, context_rnn_state, final_question, question_rnn_state = encoder_output
        encoder_loss = None
        
        if self.training and getattr(self.args, 'use_encoder_loss', None):
            encoder_loss = self.get_encoder_loss(context_rnn_state)
            
        return self.decoder(batch, self_attended_context, final_context, context_rnn_state,
                            final_question, question_rnn_state, encoder_loss, current_token_id,
                            decoder_wrapper=past_key_values,
                            expansion_factor=expansion_factor, generation_dict=generation_dict)

    def forward(self, batch, pretraining=False, current_token_id=None, past_key_values=None,
                expansion_factor=1, generation_dict=None, encoder_output=None, return_dict=False):
        """
        When training or pretraining, forward() always returns a Seq2SeqLMOutput instance
        """
        if pretraining:
            return self._pretrain_forward(batch)
        else:
            return self._normal_forward(batch, current_token_id, past_key_values, expansion_factor, generation_dict,
                                        encoder_output, return_dict)
        
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

    def prepare_inputs_for_generation(self, input_ids, attention_mask, use_cache, batch, generation_dict,
                                      encoder_output, past=None):
        expansion_factor = input_ids.shape[0] // len(batch.example_id)
        return {"batch": batch, "past_key_values": past, "current_token_id": input_ids[:, -1:],
                "expansion_factor": expansion_factor, "generation_dict": generation_dict,
                "encoder_output": encoder_output}

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
        input_ids = torch.full((batch_size, 1), self.decoder.init_idx, dtype=torch.long,
                               device=batch.context.value.device)
        
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
        generated = torch.cat(
            (generated[:, 0:1], generated[:, 1:].cpu().apply_(self.decoder.map_to_full).to(batch.context.value.device)),
            dim=1)  # map everything to full vocabulary except BOS which already is in full vocabulary

        return generated
        
class Transformer2Transformer(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        save_directory = pretrained_model_name_or_path
        model_checkpoint_file = kwargs.pop("model_checkpoint_file", None)
        args = kwargs.pop("args", None)
        device = kwargs.pop("device", None)

        full_checkpoint_path = os.path.join(save_directory, model_checkpoint_file)
        logger.info(f'Loading the model from {full_checkpoint_path}')
        model = cls(args=args)
        save_dict = torch.load(full_checkpoint_path, map_location=device)
        model.load_state_dict(save_dict['model_state_dict'])

        return model, save_dict.get('best_decascore')

    def forward(self, *input, **kwargs):
        #TODO pretraining
        if self.training:
            batch = input[0]
            pretraining = kwargs.pop("pretraining", None)

            pad = self.numericalizer._tokenizer.pad_token_id
            source_ids, source_mask, y = batch.context.value, batch.context.value!=pad, batch.answer.value
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone()
            labels[y[:, 1:] == pad] = -100
            return self.base_model.forward(source_ids, attention_mask=source_mask, decoder_input_ids=y_ids,
                                           labels=labels)

        else:
            return self.base_model.forward(**kwargs)
        
    def set_train_context_embeddings(self, trainable):
        #TODO
        pass

    def set_train_question_embeddings(self, trainable):
        #TODO
        pass

    def add_new_vocab_from_data(self, splits, resize_decoder=False):
        pass

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

        input_ids = batch.context.value
        # TODO attention_mask
        generated = self.base_model.generate(input_ids=input_ids,
                                     max_length=max_output_length,
                                     min_length=2, # generate at least one token after BOS
                                     bos_token_id=self.numericalizer._tokenizer.bos_token_id,
                                     pad_token_id=self.numericalizer._tokenizer.pad_token_id,
                                     early_stopping=True,
                                     num_return_sequences=num_outputs,
                                     repetition_penalty=repetition_penalty,
                                     temperature=temperature,
                                     eos_token_id=self.numericalizer._tokenizer.eos_token_id,
                                     top_k=top_k,
                                     top_p=top_p,
                                     num_beams=num_beams,
                                     no_repeat_ngram_size=no_repeat_ngram_size,
                                     do_sample=do_sample,
                                    )

        return generated

class Bart(Transformer2Transformer):

    def __init__(self, config=None, *inputs, **kwargs):
        super().__init__()
        assert 'args' in kwargs
        self.args = kwargs['args']
        self.base_model = BartForConditionalGeneration.from_pretrained(self.args.seq2seq_decoder,
                                                                       cache_dir=self.args.embeddings)
        self.numericalizer = BartNumericalizer(self.args.seq2seq_decoder)


class MBart(Transformer2Transformer):
    
    def __init__(self, config=None, *inputs, **kwargs):
        super().__init__()
        assert 'args' in kwargs
        self.args = kwargs['args']
        self.base_model = MBartForConditionalGeneration.from_pretrained(self.args.seq2seq_decoder,
                                                                        cache_dir=self.args.embeddings)
        self.numericalizer = MBartNumericalizer(self.args.seq2seq_decoder)


class MT5(Transformer2Transformer):

    def __init__(self, config=None, *inputs, **kwargs):
        super().__init__()
        assert 'args' in kwargs
        self.args = kwargs['args']
        self.base_model = MT5ForConditionalGeneration.from_pretrained(self.args.seq2seq_decoder,
                                                                      cache_dir=self.args.embeddings)
        self.numericalizer = MT5Numericalizer(self.args.seq2seq_decoder)
