#
# Copyright (c) 2020, The Board of Trustees of the Leland Stanford Junior University
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
import os
import shutil

from .data_utils.embeddings import load_embeddings
from .util import load_config_json

logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument('--path', required=True,
                        help='the model training directory to export')
    parser.add_argument('--embeddings', default='.embeddings/', type=str, help='where to load embeddings from')
    parser.add_argument('--checkpoint_name', default='best.pth',
                        help='Checkpoint file to use (relative to --path, defaults to best.pth)')
    parser.add_argument('-o', '--output', required=True,
                        help='the directory where to export into')


def main(args):
    os.makedirs(args.output, exist_ok=True)
    load_config_json(args)

    # we need to load the embeddings to get to the correct numericalizer class
    # this is somewhat unfortunate but acceptable
    numericalizer, _, _ = load_embeddings(args.embeddings, args.encoder_embeddings,
                                          args.decoder_embeddings,
                                          args.max_generative_vocab,
                                          logger)

    # load the numericalizer from the model training directory, and immediately save it in the export directory
    # this will copy over all the necessary vocabulary and config files that the numericalizer needs
    numericalizer.load(args.path)
    numericalizer.save(args.output)

    # now copy over the config.json and checkpoint file
    for fn in ['config.json', args.checkpoint_name]:
        src = os.path.join(args.path, fn)
        dst = os.path.join(args.output, fn)
        shutil.copyfile(src, dst)

    logger.info(f'Successfully exported model from {args.path} to {args.output}')