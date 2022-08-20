#
# Copyright (c) 2022, The Board of Trustees of the Leland Stanford Junior University
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
from pprint import pformat

import torch

from . import models
from .arguments import check_and_update_generation_args
from .tasks.registry import get_tasks
from .util import get_devices, load_config_file_to_args, set_seed

logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--tasks', dest='task_names', nargs='+', help='task names for prediction')

    parser.add_argument(
        '--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used (multi-gpu currently WIP)'
    )
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--embeddings', default='.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument(
        '--checkpoint_name', default='best.pth', help='Checkpoint file to use (relative to --path, defaults to best.pth)'
    )
    parser.add_argument('--eval_dir', type=str, help='use this directory to store eval results')

    parser.add_argument('--database_dir', type=str, help='Database folder containing all relevant files')
    parser.add_argument('--src_locale', help='locale tag of the input language to parse')
    parser.add_argument('--tgt_locale', help='locale tag of the target language to generate')
    parser.add_argument('--inference_name', default='nlp', help='name used by kfserving inference service, alphanumeric only')

    # These are generation hyperparameters. Each one can be a list of values in which case, we generate `num_outputs` outputs for each set of hyperparameters.
    parser.add_argument("--num_outputs", type=int, nargs='+', default=[1], help='number of sequences to output per input')
    parser.add_argument("--temperature", type=float, nargs='+', default=[0.0], help="temperature of 0 implies greedy sampling")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        nargs='+',
        default=[1.0],
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[1.0], help='1.0 disables top-p filtering')
    parser.add_argument("--num_beams", type=int, nargs='+', default=[1], help='1 disables beam seach')
    parser.add_argument("--num_beam_groups", type=int, nargs='+', default=[1], help='1 disables diverse beam seach')
    parser.add_argument("--diversity_penalty", type=float, nargs='+', default=[0.0], help='0 disables diverse beam seach')
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        nargs='+',
        default=[0],
        help='ngrams of this size cannot be repeated in the output. 0 disables it.',
    )
    parser.add_argument('--max_output_length', default=150, type=int, help='maximum output length for generation')


class DialogueLoop(object):
    def __init__(self, args, model):
        self.model = model
        self.args = args

    def run(self):
        task = list(get_tasks(self.args.task_names, self.args).values())[0]
        self.model.add_new_vocab_from_data([task])
        self.model.set_generation_output_options([task])

        with torch.no_grad():
            self.model.interact_e2e_dialogues(task, eval_dir=self.args.eval_dir)


def init(args):
    set_seed(args)

    devices = get_devices()
    device = devices[0]  # server only runs on a single device

    load_config_file_to_args(args)
    check_and_update_generation_args(args)

    if not args.src_locale:
        args.src_locale = args.eval_src_languages
    if not args.tgt_locale:
        args.tgt_locale = args.eval_tgt_languages

    if args.eval_dir:
        os.makedirs(args.eval_dir, exist_ok=True)

    Model = getattr(models, args.model)
    model, _ = Model.load(
        args.path,
        model_checkpoint_file=args.checkpoint_name,
        args=args,
        device=device,
        src_lang=args.src_locale,
        tgt_lang=args.tgt_locale,
    )
    model.to(device)
    model.eval()
    logger.info(f'Arguments:\n{pformat(vars(args))}')
    logger.info(f'Loading from {args.best_checkpoint}')

    return model


def main(args):
    model = init(args)
    loop = DialogueLoop(args, model)
    loop.run()
