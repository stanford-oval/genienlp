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

import itertools

from ..util import set_seed
from decanlp.utils.model_utils import init_model
from .common import *

import logging

_logger = logging.getLogger(__name__)

class MultiLingualTranslationModel(nn.Module):

    def __init__(self, field, args):
        super().__init__()

        # self.field = field
        self.args = args
        # self.pad_idx = self.field.vocab.stoi[self.field.pad_token]
        self.device = set_seed(args)


        for name in ['machine_translation', 'semantic_parser', 'thingtalk_machine_translation']:

            model_dict = torch.load(os.path.join(args.saved_models, f'{name}/best.pth'), map_location=self.device)
            model_field = model_dict['field']
            model = init_model(args, model_field, _logger, args.world_size, self.device)
            model.load_state_dict(model_dict['model_state_dict'])
            # model_opt_dict = torch.load(os.path.join(args.saved_models, f'{model}/best_optim.pth'), map_location=self.device)
            # model_opt = init_opt(args, model)
            # model_opt.load_state_dict(model_opt_dict)

            setattr(self, f'{name}_field', model_field)
            setattr(self, f'{name}_model', model)
            # setattr(self, f'{name}_opt', model_opt)


    # def set_embeddings(self, embeddings):
    #     self.encoder_embeddings.set_embeddings(embeddings)
    #     if self.decoder_embeddings is not None:
    #         self.decoder_embeddings.set_embeddings(embeddings)

    def forward(self, batch, iteration):
        context, context_lengths, context_limited, context_tokens     = batch.context,  batch.context_lengths,  batch.context_limited, batch.context_tokens
        question, question_lengths, question_limited, question_tokens = batch.question, batch.question_lengths, batch.question_limited, batch.question_tokens
        answer, answer_lengths, answer_limited, answer_tokens         = batch.answer,   batch.answer_lengths,   batch.answer_limited, batch.answer_tokens
        oov_to_limited_idx, limited_idx_to_full_idx  = batch.oov_to_limited_idx, batch.limited_idx_to_full_idx

        batch_size = len(question_tokens)

        # running through Farsi-English MT
        question = 'Translate from Farsi to English'.split()
        question_tokens_single = [self.machine_translation_field.init_token] + question + [self.machine_translation_field.eos_token]
        question_tokens = list(itertools(question_tokens_single, batch_size))
        context_numericalized = [[self.machine_translation_field.decoder_stoi[sentence[time]] for sentence in context_tokens] for time in range(len(context_tokens[0]))]
        question_numericalized = [[self.machine_translation_field.decoder_stoi[sentence[time]] for sentence in question_tokens] for time in range(len(question_tokens[0]))]
        setattr(batch, context, context_numericalized)
        setattr(batch, question, question_numericalized)

        self.machine_translation_model.eval()
        _, greedy_output_from_MT = self.machine_translation_model(batch, iteration)


        # numericalize input to semantic parser
        question = 'Translate from English to ThingTalk'.split()
        question_tokens_single = [self.semantic_parser_field.init_token] + question + [self.semantic_parser_field.eos_token]
        question_tokens = list(itertools(question_tokens_single, batch_size))

        context_numericalized = [[self.semantic_parser_field.decoder_stoi[sentence[time]] for sentence in greedy_output_from_MT] for time in range(len(greedy_output_from_MT[0]))]
        question_numericalized = [[self.semantic_parser_field.decoder_stoi[sentence[time]] for sentence in question_tokens] for time in range(len(question_tokens[0]))]
        setattr(batch, context, context_numericalized)
        setattr(batch, question, question_numericalized)

        self.semantic_parser_model.eval()
        _, greedy_output_from_SP = self.semantic_parser_model(batch, iteration)

        # numericalize input to thingtalk machine translation
        question = 'Translate from English to Farsi'.split()
        question_tokens_single = [self.semantic_parser_field.init_token] + question + [self.semantic_parser_field.eos_token]
        question_tokens = list(itertools(question_tokens_single, batch_size))

        context_numericalized = [[self.thingtalk_machine_translation_field.decoder_stoi[sentence[time]] for sentence in greedy_output_from_SP] for time in range(len(greedy_output_from_SP[0]))]
        question_numericalized = [[self.thingtalk_machine_translation_field.decoder_stoi[sentence[time]] for sentence in question_tokens] for time in range(len(question_tokens[0]))]
        answer_numericalized = [[self.thingtalk_machine_translation_field.decoder_stoi[sentence[time]] for sentence in answer_tokens] for time in range(len(answer_tokens[0]))]
        setattr(batch, context, context_numericalized)
        setattr(batch, question, question_numericalized)
        setattr(batch, answer, answer_numericalized)

        if self.training:

            self.thingtalk_machine_translation_model.train()
            loss, _ = self.thingtalk_machine_translation_model(batch, iteration)

            return loss, None

        else:

            self.thingtalk_machine_translation_model.eval()
            _, greedy_output_from_thingtalk_MT = self.thingtalk_machine_translation_model(batch, iteration)

            return None, greedy_output_from_thingtalk_MT

