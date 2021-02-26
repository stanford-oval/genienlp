#
# Copyright (c) 2020 The Board of Trustees of the Leland Stanford Junior University
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

import logging
from typing import List
import torch

from torch.tensor import Tensor
from transformers import AutoModelForSeq2SeqLM, AutoConfig

from ..data_utils.numericalizer import TransformerNumericalizer
from ..util import get_mbart_lang
from .base import GenieModel
from ..util import ConfidenceFeatures
from .common import LabelSmoothingCrossEntropy

logger = logging.getLogger(__name__)


class TransformerSeq2Seq(GenieModel):
    def __init__(self, config=None, *inputs, args, tasks, vocab_sets, save_directory=None, **kwargs):
        config = AutoConfig.from_pretrained(args.pretrained_model, cache_dir=args.embeddings)
        super().__init__(config)
        self.args = args
        args.dimension = config.d_model
        assert args.dimension == config.hidden_size, "In HuggingFace transformers 4.1 Seq2Seq models, `hidden_size` and `d_model` are the same parameter"
        self._is_bart_large = self.args.pretrained_model == 'facebook/bart-large'
        self._is_mbart = 'mbart' in self.args.pretrained_model
        
        if save_directory is not None:
            self.model = AutoModelForSeq2SeqLM.from_config(config)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.pretrained_model,
                                                               cache_dir=self.args.embeddings)

        self.numericalizer = TransformerNumericalizer(self.args.pretrained_model, args, max_generative_vocab=None)

        self.numericalizer.get_tokenizer(save_directory)
        
        if self._is_mbart:
            self._adjust_mbart(kwargs.get('locale', 'en'))

        self.init_vocab_from_data(vocab_sets, tasks, save_directory)
        self.model.resize_token_embeddings(self.numericalizer.num_tokens)

        if args.dropper_ratio > 0:
            # lazy import since dropper is an optional dependency 
            from loss_dropper import LossDropper
            self.dropper = LossDropper(dropc=args.dropper_ratio, min_count=args.dropper_min_count)
        else:
            self.dropper = None

        self.criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
        self.error_classifier = torch.nn.Linear(in_features=self.args.dimension, out_features=1, bias=True)
        self.error_classifier_criterion = torch.nn.BCELoss()
            
            
    def _adjust_mbart(self, lang):
        # We need to set language id for mBART models as it is used during tokenization and generation
        # For now we only support single language training and evaluation with mbart models
        lang_id = get_mbart_lang(lang)
        self.numericalizer._tokenizer.set_src_lang_special_tokens(lang_id)
        self.model.config.decoder_start_token_id = self.numericalizer._tokenizer.cur_lang_code
        

    def add_new_vocab_from_data(self, tasks, resize_decoder=False):
        super().add_new_vocab_from_data(tasks, resize_decoder)
        self.model.resize_token_embeddings(self.numericalizer.num_tokens)
    
    def apply_error_classifier(self, decoder_hidden_states, answer_length):
        error_classifier_input = torch.gather(decoder_hidden_states[-1], dim=1, index=(answer_length-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.args.dimension)).squeeze(1) # (batch_size, hidden_size)
        # for i in range(error_classifier_input.shape[0]):
        #     mismatch = torch.abs(torch.sum(error_classifier_input[i,:] - decoder_hidden_states[-1][i][answer_length[i]-1, :])).item()
        #     assert mismatch < 1e-4
        error_classifier_output = self.error_classifier(error_classifier_input)
        return error_classifier_output

    def forward(self, *input, **kwargs):
        if self.training:
            batch = input[0]
            # print('batch.bad_answer = ', batch.bad_answer)
            # print('batch.answer = ', batch.answer)
            answer = batch.answer.value
            answer_length = batch.answer.length
            bad_answer = batch.bad_answer.value
            bad_answer_length = batch.bad_answer.length
            if self._is_bart_large:
                # remove BOS from the answer to BART-Large because BART-Large was not trained to predict BOS
                # (unlike BART-Base or mBART)
                #
                # NOTE: various people at Huggingface and elsewhere have tried to conclusively ascertain
                # whether BOS should be there or not, and the answer seems to be that BOS should not be there
                # at all, either in input or in the output
                # but empirically, BOS in the input works slightly better, pehraps because our sentences start
                # with a lowercase letter, so we leave it
                answer = answer[:, 1:].contiguous()
                answer_length = answer_length - 1
                bad_answer = bad_answer[:, 1:].contiguous()
                bad_answer_length = bad_answer_length - 1

            # this has several differences compared to what `transformers` Seq2Seq models do:
            # (1) pad tokens are ignored in all loss calculations
            # (2) loss is averaged over sequence lengths first, then over the batch size. This way,
            # longer sequences in the batch do not drown shorter sequences.
            # (3) if `args.dropper_ratio > 0.0`, will perform Loss Truncation
            # (4) if `args.label_smoothing > 0.0`, will add label smoothing term to loss

            # calculate the classification loss for correct parses
            outputs = self.model(batch.context.value, labels=answer, attention_mask=(batch.context.value!=self.numericalizer.pad_id), output_hidden_states=True)
            # print('answer_length = ', answer_length)
            # print('decoder_hidden_states = ', outputs.decoder_hidden_states[-1].shape)
            error_classifier_output_1 = self.apply_error_classifier(outputs.decoder_hidden_states, answer_length)
            # print('error_classifier_output_1 = ', error_classifier_output_1.shape)
            error_classifier_loss_1 = self.error_classifier_criterion(torch.sigmoid(error_classifier_output_1), target=torch.ones_like(error_classifier_output_1))
            # print('error_classifier_loss_1 = ', error_classifier_loss_1)

            # calculate the teacher-forcing loss
            batch_size, vocab_size = outputs.logits.shape[0], outputs.logits.shape[2]
            loss = self.criterion(outputs.logits.view(-1, vocab_size), target=answer.view(-1), ignore_index=self.numericalizer.pad_id)
            loss = loss.view(batch_size, -1) # (batch_size, sequence_length)
            loss = loss.sum(dim=1) / answer_length # accounts for the case where BOS is removed
            if self.dropper is not None:
                dropper_mask = self.dropper(loss)
                loss = loss * dropper_mask
            loss = loss.mean() # average over the batch size

            # calculate the classification loss for bad parses
            outputs = self.model(batch.context.value, labels=bad_answer, attention_mask=(batch.context.value!=self.numericalizer.pad_id), output_hidden_states=True)
            error_classifier_output_2 = self.apply_error_classifier(outputs.decoder_hidden_states, bad_answer_length)
            # print('error_classifier_output_2 = ', error_classifier_output_2.shape)
            error_classifier_loss_2 = self.error_classifier_criterion(torch.sigmoid(error_classifier_output_2), target=torch.zeros_like(error_classifier_output_2))
            # print('error_classifier_loss_2 = ', error_classifier_loss_2)

            # shuffle_vector = torch.randint(2, size=(error_classifier_output_1.shape[0], 1), device=error_classifier_output_1.device)
            # error_classifier_output_1 = shuffle_vector*error_classifier_output_1 + (1-shuffle_vector)*error_classifier_output_2
            # error_classifier_output_2 = (1-shuffle_vector)*error_classifier_output_1 + shuffle_vector*error_classifier_output_2
            
            # error_classifier_output = torch.cat([error_classifier_output_1, error_classifier_output_2], dim=1)
            # print('error_classifier_output = ', error_classifier_output.shape)
            # error_detection_loss = self.error_classifier_criterion(error_classifier_output, target=torch.zeros_like(bad_answer_length))
            # zero_centered_penalty = torch.linalg.norm(error_classifier_output[:,0]+error_classifier_output[:,1]) / batch_size
            # print('zero_centered_penalty = ', zero_centered_penalty)

            outputs.loss = loss + error_classifier_loss_1 + error_classifier_loss_2 # replace the loss calculated by `transformers` with the new loss
            return outputs
        else:
            return self.model(**kwargs)

    def generate(self,
                 batch,
                 max_output_length,
                 num_outputs,
                 temperature,
                 repetition_penalty,
                 top_k,
                 top_p,
                 num_beams,
                 num_beam_groups,
                 diversity_penalty,
                 no_repeat_ngram_size,
                 do_sample
                 ):
        
        decoder_start_token_id = None
        if self._is_mbart:
            decoder_start_token_id = self.model.config.decoder_start_token_id

        input_ids = batch.context.value
        # when attention_mask is not provided to generate(), it will default to masking pad tokens, which is the correct thing
        generated = self.model.generate(input_ids=input_ids,
                                        max_length=max_output_length,
                                        min_length=2, # generate at least one token after BOS
                                        bos_token_id=self.numericalizer.init_id,
                                        pad_token_id=self.numericalizer.pad_id,
                                        early_stopping=False,
                                        num_return_sequences=num_outputs,
                                        repetition_penalty=repetition_penalty,
                                        temperature=temperature,
                                        eos_token_id=self.numericalizer.eos_id,
                                        top_k=top_k,
                                        top_p=top_p,
                                        num_beams=num_beams,
                                        num_beam_groups=num_beam_groups,
                                        diversity_penalty=diversity_penalty,
                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                        do_sample=do_sample,
                                        decoder_start_token_id=decoder_start_token_id,
                                        )
        
        return generated


    def confidence_features(self, batch, predictions, mc_dropout_num=0) -> List[ConfidenceFeatures]:
        """
        predictions: Tensor of shape (batch_size, output_length)
        mc_droput_num: number of Monte Carlo samples used for the MC Dropout method. 0 disables MC dropout.
        """
        
        batch_size = predictions.shape[0]
        repetition_factor = batch_size//batch.context.value.shape[0]
        input_ids = batch.context.value.repeat_interleave(repetition_factor, dim=0) # repeat to account for multiple predictions per input

        prediction_lengths = self.get_length(predictions)

        pad_token_id = self.numericalizer.pad_id
        attention_mask = self.model._prepare_attention_mask_for_generation(input_ids=input_ids, pad_token_id=pad_token_id, eos_token_id=self.numericalizer.eos_id)
        truncated_predictions = predictions[:, 1:] # remove the BOS token since it is not actually being generated

        assert not self.training, 'Model should be in eval() mode before generation can start.'

        batch_nodrop_logits = []
        batch_nodrop_probs = []
        batch_nodrop_entropies = []
        correct_logits = []
        # batch_nodrop_top1_probs = []
        # batch_nodrop_top1_idx = []
        # batch_nodrop_top2_probs = []
        # batch_nodrop_top2_idx = []
        # print('input_ids[-1] = ', input_ids[-1])
        # print('attention_mask[-1] = ', attention_mask[-1])
        # print('prediction_lengths[-1] = ', prediction_lengths[-1])
        # print('decoder_input_ids[-1] = ', predictions[-1])
        outputs = self.model(input_ids=input_ids, decoder_input_ids=predictions, attention_mask=attention_mask, return_dict=True, use_cache=False, output_hidden_states=True)
        error_classifier_output = self.apply_error_classifier(outputs.decoder_hidden_states, prediction_lengths+1) # +1 is necessary here but not during training
        correct_logits.extend(error_classifier_output.squeeze(0).tolist())
        # print('error_classifier_output = ', error_classifier_output)

        nodrop_logits = outputs.logits[:, :-1, :] # remove the last probability distribution which is for the token after EOS
        for i in range(batch_size):
            batch_nodrop_logits.append(nodrop_logits[i].gather(dim=1, index=truncated_predictions[i].view(-1, 1)).view(-1)[:prediction_lengths[i]])
            probs = torch.softmax(nodrop_logits[i], dim=1)
            batch_nodrop_probs.append(probs.gather(dim=1, index=truncated_predictions[i].view(-1, 1)).view(-1)[:prediction_lengths[i]])
            batch_nodrop_entropies.append(-torch.sum(torch.log(probs)*probs, dim=1)[:prediction_lengths[i]])
            # sorted_probs = probs.sort(dim=1)
            # batch_nodrop_top1_probs.append(sorted_probs.values[:, -1][:prediction_lengths[i]])
            # batch_nodrop_top2_probs.append(sorted_probs.values[:, -2][:prediction_lengths[i]])
            # batch_nodrop_top1_idx.append(sorted_probs.indices[:, -1][:prediction_lengths[i]])
            # batch_nodrop_top2_idx.append(sorted_probs.indices[:, -2][:prediction_lengths[i]])
        
        # activate dropout layers
        self.train()

        batch_drop_logits = [[] for _ in range(batch_size)]
        batch_drop_probs = [[] for _ in range(batch_size)]
        # batch_drop_top1_probs = [[] for _ in range(batch_size)]
        # batch_drop_top2_probs = [[] for _ in range(batch_size)]
        for _ in range(mc_dropout_num):
            outputs = self.model(input_ids=input_ids, decoder_input_ids=predictions, attention_mask=attention_mask, return_dict=True, use_cache=False)
            drop_logits = outputs.logits[:, :-1, :] # remove the last probability distribution which is for the token after EOS
            for i in range(batch_size):
                batch_drop_logits[i].append((drop_logits[i].gather(dim=1, index=truncated_predictions[i].view(-1, 1)).view(-1))[:prediction_lengths[i]])
                softmax = torch.softmax(drop_logits[i], dim=1)
                drop_probs = softmax.gather(dim=1, index=truncated_predictions[i].view(-1, 1)).view(-1)[:prediction_lengths[i]]
                batch_drop_probs[i].append(drop_probs)
                # batch_drop_top1_probs[i].append(softmax.gather(dim=1, index=batch_nodrop_top1_idx[i].view(-1, 1)).view(-1)[:prediction_lengths[i]])
                # batch_drop_top2_probs[i].append(softmax.gather(dim=1, index=batch_nodrop_top2_idx[i].view(-1, 1)).view(-1)[:prediction_lengths[i]])

        confidence_features = []
        for i in range(batch_size):
            confidence_features.append(
                        ConfidenceFeatures(drop_logits=batch_drop_logits[i] if mc_dropout_num > 0 else None,
                                         drop_probs=batch_drop_probs[i] if mc_dropout_num > 0 else None,
                                        #  drop_top1_probs=batch_drop_top1_probs[i] if mc_dropout_num > 0 else None,
                                        #  drop_top2_probs=batch_drop_top2_probs[i] if mc_dropout_num > 0 else None,
                                         gold_answer=batch.answer.value[i//repetition_factor][:batch.answer.length[i//repetition_factor]],
                                         prediction=predictions[i][:prediction_lengths[i]+1],  # +1 to include EOS
                                         nodrop_logits=batch_nodrop_logits[i][:prediction_lengths[i]],
                                         nodrop_probs=batch_nodrop_probs[i][:prediction_lengths[i]],
                                        #  nodrop_top1_probs=batch_nodrop_top1_probs[i][:prediction_lengths[i]],
                                        #  nodrop_top2_probs=batch_nodrop_top2_probs[i][:prediction_lengths[i]],
                                         nodrop_entropies=batch_nodrop_entropies[i][:prediction_lengths[i]],
                                         context=batch.context.value[i//repetition_factor][:batch.context.length[i//repetition_factor]],
                                         correct_logit=correct_logits[i],
                                         ))

        # return the model back to its previous state
        self.eval()
        
        return confidence_features

    def get_length(self, prediction:Tensor):
        # skip the first token, because BOS is the same as EOS for some models
        prediction = prediction[:, 1:]

        # add EOS at the end in case the prediction doesn't have any
        prediction = torch.cat([prediction, torch.ones((prediction.shape[0], 1), dtype=torch.long, device=prediction.device)*self.numericalizer.eos_id], dim=1)

        # find the index of the first eos
        first_eos_one_hot = (torch.cumsum((prediction == self.numericalizer.eos_id).long(), dim=1) == 1) & (prediction == self.numericalizer.eos_id)
        first_eos = first_eos_one_hot.nonzero(as_tuple=False)[:, 1] + 1 # +1 to account for the first token that we ignored
        return first_eos