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

from argparse import ArgumentParser
import logging
import torch

from .util import load_config_json, set_seed, init_devices
from .tasks.registry import get_tasks
from .data_utils.embeddings import load_embeddings
from . import models


logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument('--path', required=True)
    parser.add_argument('--tasks', default=['almond', 'squad', 'iwslt.en.de', 'cnn_dailymail', 'multinli.in.out', 'sst','srl', 'zre', 'woz.en', 'wikisql', 'schema'], dest='task_names', nargs='+')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--embeddings', default='./.embeddings', type=str, help='where to load embeddings from')
    parser.add_argument('--checkpoint_name', default='best.pth',
                        help='Checkpoint file to use (relative to --path, defaults to best.pth)')
    parser.add_argument('-o', '--output', required=True,
                        help='Location where the compiled module should be saved')


def main(args):
    load_config_json(args)
    set_seed(args)
    args.tasks = get_tasks(args.task_names, args)
    logger.info(f'Loading from {args.best_checkpoint}')

    numericalizer, context_embeddings, question_embeddings, decoder_embeddings = \
        load_embeddings(args.embeddings, args.context_embeddings, args.question_embeddings, args.decoder_embeddings,
                        args.max_generative_vocab, logger)
    numericalizer.load(args.path)
    for emb in set(context_embeddings + question_embeddings + decoder_embeddings):
        emb.init_for_vocab(numericalizer.vocab)

    logger.info(f'Initializing Model')
    Model = getattr(models, args.model)
    model = Model.from_pretrained(args.path,
                                  numericalizer=numericalizer,
                                  context_embeddings=context_embeddings,
                                  question_embeddings=question_embeddings,
                                  decoder_embeddings=decoder_embeddings,
                                  args=args)

    compiled = torch.jit.script(model)
    compiled.save(args.output)
