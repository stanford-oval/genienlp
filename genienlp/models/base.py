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

import torch
import logging
import os

from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class GenieModel(PreTrainedModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        save_directory = pretrained_model_name_or_path
        model_checkpoint_file = kwargs.pop("model_checkpoint_file", None)
        args = kwargs.pop("args", None)
        device = kwargs.pop("device", None)
        tasks = kwargs.pop("tasks", None)
        vocab_sets = kwargs.pop("vocab_sets", None)

        full_checkpoint_path = os.path.join(save_directory, model_checkpoint_file)
        logger.info(f'Loading the model from {full_checkpoint_path}')
        model = cls(args=args, tasks=tasks, vocab_sets=vocab_sets, save_directory=save_directory)
        save_dict = torch.load(full_checkpoint_path, map_location=device)
        model.load_state_dict(save_dict['model_state_dict'])

        return model, save_dict.get('best_decascore')

    def init_vocab_from_data(self, vocab_sets, tasks, save_directory=None):
        if save_directory is not None:
            logger.info(f'Loading the accompanying numericalizer from {save_directory}')
            self.numericalizer.load(save_directory)
        else:
            logger.info(f'Building vocabulary')
            self.numericalizer.build_vocab(vocab_sets, tasks)

    def add_new_vocab_from_data(self, tasks, resize_decoder=False):
        old_num_tokens = self.numericalizer.num_tokens
        self.numericalizer.grow_vocab(tasks)
        if self.numericalizer.num_tokens > old_num_tokens:
            logger.info(f'Vocabulary has expanded to {self.numericalizer.num_tokens} tokens')
        self.bart.resize_token_embeddings(self.numericalizer.num_tokens)
        if resize_decoder:
            self.decoder.decoder_embeddings.resize_embedding(self.numericalizer.num_tokens)