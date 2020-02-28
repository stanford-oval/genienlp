#
# Copyright (c) 2018, The Board of Trustees of the Leland Stanford Junior University
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
'''
Created on Mar 3, 2019

@author: gcampagn
'''

import torch
import json
import os
import logging

logger = logging.getLogger(__name__)


class Saver(object):
    '''
    Wrap pytorch's save functionality into an interface similar to tensorflow.train.Saver
    
    In particular, this class takes care of automatically cleaning up old checkpoints,
    and creating checkpoint files to keep track of which saves are valid and which are not.
    '''

    def __init__(self, savedir, max_to_keep=5):
        self._savedir = savedir
        self._max_to_keep = max_to_keep
        assert max_to_keep >= 1

        self._loaded_last_checkpoints = False
        self._latest_checkpoint = None
        self._all_checkpoints = None

    def _maybe_load_last_checkpoints(self):
        if self._loaded_last_checkpoints:
            return

        try:
            with open(os.path.join(self._savedir, 'checkpoint.json')) as fp:
                data = json.load(fp)
                self._loaded_last_checkpoints = True
                self._all_checkpoints = data['all']
                self._latest_checkpoint = data['latest']
        except FileNotFoundError:
            self._loaded_last_checkpoints = True
            self._all_checkpoints = []
            self._latest_checkpoint = None

    def save(self, save_model_state_dict, save_opt_state_dict, global_step):
        self._maybe_load_last_checkpoints()

        model_name = 'iteration_' + str(global_step) + '.pth'
        opt_name = 'iteration_' + str(global_step) + '_optim.pth'

        self._latest_checkpoint = model_name
        self._all_checkpoints.append(model_name)
        if len(self._all_checkpoints) > self._max_to_keep:
            try:
                todelete = self._all_checkpoints.pop(0)
                os.unlink(os.path.join(self._savedir, todelete))
                opt_todelete = todelete.rsplit('.', 1)[0] + '_optim.' + todelete.rsplit('.', 1)[1]
                os.unlink(os.path.join(self._savedir, opt_todelete))
            except (OSError, IOError) as e:
                logging.warning('Failed to delete old checkpoint: %s', e)
        torch.save(save_model_state_dict, os.path.join(self._savedir, model_name))
        torch.save(save_opt_state_dict, os.path.join(self._savedir, opt_name))
        with open(os.path.join(self._savedir, 'checkpoint.json'), 'w') as fp:
            json.dump(dict(all=self._all_checkpoints, latest=self._latest_checkpoint), fp)
