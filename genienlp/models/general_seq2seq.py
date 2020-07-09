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
from torch.nn import functional as F
from torch.utils.data import DataLoader

ENCODERS = {
    'MQANEncoder': MQANEncoder,
    'BiLSTM': BiLSTMEncoder,
    'Identity': IdentityEncoder,
    'Coattention': CoattentionEncoder,
}
DECODERS = {
    'MQANDecoder': MQANDecoder
}


class Seq2Seq(torch.nn.Module):
    def __init__(self, numericalizer, args, context_embeddings, question_embeddings, decoder_embeddings):
        super().__init__()
        self.args = args

        self.numericalizer = numericalizer
        self.encoder = ENCODERS[args.seq2seq_encoder](numericalizer, args, context_embeddings, question_embeddings)
        self.decoder = DECODERS[args.seq2seq_decoder](numericalizer, args, decoder_embeddings)

        if self.args.pretrain_context > 0:
            self.context_pretrain_lm_head = torch.nn.Linear(self.args.dimension, numericalizer.num_tokens)
        
        self.EWC_task_count = 0
        self.ewc_gamma = 1.
        # self.ewc = getattr(self.args, 'use_ewc', False)
        # # hyperparam: how strong to weigh the EWC loss from the previous task
        # self.ewc_lambda =  getattr(self.args, 'ewc_lambda', 0)

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

    def _normal_forward(self, batch):
        self_attended_context, final_context, context_rnn_state, final_question, question_rnn_state = \
            self.encoder(batch)
        encoder_loss = None
        if getattr(self.args, 'use_encoder_loss', None) and self.training:
            encoder_loss = self.get_encoder_loss(context_rnn_state)
        return self.decoder(batch, self_attended_context, final_context, context_rnn_state,
                            final_question, question_rnn_state, encoder_loss)

    def forward(self, batch, iteration, pretraining=False):
        # print('batch = ', batch)
        if pretraining:
            loss, predictions = self._pretrain_forward(batch)

            if self.args.use_ewc:
                # add EWC loss
                ewc_loss = self.ewc_loss()
                if self.args.ewc_lambda>0:
                    loss += self.ewc_lambda * ewc_loss

            return loss, predictions
        else:
            loss, predictions = self._normal_forward(batch)

            if self.args.use_ewc:
                # add EWC loss
                ewc_loss = self.ewc_loss()
                if self.args.ewc_lambda>0:
                    loss += self.ewc_lambda * ewc_loss

            return loss, predictions
        
        
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

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count>0:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_EWC_prev_task'.format(n))
                    fisher = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    # since "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                    fisher = self.ewc_gamma*fisher
                    # Calculate EWC-loss
                    losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())
    
    def estimate_fisher(self, dataset):
        '''Estimate diagonal of Fisher Information matrix.'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Estimate the FI-matrix for batches of size 1
        data_loader = self.fisher_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda())
        for index, (x, y) in enumerate(data_loader):

            # run forward pass of model
            x = x.to(self._device())
            output = self(x)

            # use predicted label to calculate loglikelihood
            label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # mode (MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task'.format(n),
                                        p.detach().clone())
                # precision (approximated by diagonal Fisher Information matrix)
                if self.EWC_task_count==1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.ewc_gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher'.format(n),
                                        est_fisher_info[n])

        # set task-count to 1 to indicate EWC-loss can be calculated
        self.EWC_task_count = 1

        # set model back to initial mode
        self.train(mode=mode)

    def fisher_data_loader(dataset, batch_size, cuda=False, collate_fn=None, drop_last=False):
        '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

        # Create and return the <DataLoader>-object
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=(collate_fn or default_collate), drop_last=drop_last,
            **({'num_workers': 0, 'pin_memory': True} if cuda else {})
    )
