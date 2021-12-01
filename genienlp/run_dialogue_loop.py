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
import copy
import logging
from pprint import pformat

import torch
from BiToD.templates.generate_template_response import TemplateResponseGenerator

from genienlp.dial_validate import generate_with_seq2seq_model_for_dialogue_interactive

from . import models
from .arguments import check_and_update_generation_args
from .tasks.registry import get_tasks
from .util import get_devices, load_config_json, set_seed

logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument('--e2e_path', type=str, required=True)
    parser.add_argument('--nlg_path', type=str)
    parser.add_argument(
        '--nlg_type', type=str, choices=['neural', 'template-translated', 'template-human'], default='template'
    )

    parser.add_argument(
        '--devices', default=[0], nargs='+', type=int, help='a list of devices that can be used (multi-gpu currently WIP)'
    )
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--embeddings', default='.embeddings', type=str, help='where to save embeddings.')
    parser.add_argument(
        '--checkpoint_name', default='best.pth', help='Checkpoint file to use (relative to --path, defaults to best.pth)'
    )

    parser.add_argument('--database_dir', type=str, help='Database folder containing all relevant files')
    parser.add_argument('--src_locale', default='en', help='locale tag of the input language to parse')
    parser.add_argument('--tgt_locale', default='en', help='locale tag of the target language to generate')
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
    def __init__(self, e2e_model, nlg_model):
        self.e2e_model = e2e_model
        self.nlg_model = nlg_model

    def run(self):
        e2e_task = list(get_tasks(['bitod'], self.e2e_model.args).values())[0]
        self.e2e_model.add_new_vocab_from_data([e2e_task])
        self.e2e_model.set_generation_output_options([e2e_task])

        if self.e2e_model.args.nlg_type == 'neural':
            nlg_task = list(get_tasks(['bitod_nlg'], self.nlg_model.args).values())[0]
            self.nlg_model.add_new_vocab_from_data([nlg_task])
            self.nlg_model.set_generation_output_options([nlg_task])
        else:
            nlg_task = None

        with torch.no_grad():
            generate_with_seq2seq_model_for_dialogue_interactive(
                self.e2e_model,
                self.nlg_model,
                e2e_task,
                nlg_task,
            )


def init(args):
    set_seed(args)

    devices = get_devices()
    device = devices[0]  # server only runs on a single device

    e2e_args = copy.deepcopy(args)
    e2e_args.path = args.e2e_path
    load_config_json(e2e_args)
    check_and_update_generation_args(e2e_args)

    E2EModel = getattr(models, e2e_args.model)
    e2e_model, _ = E2EModel.load(
        e2e_args.path,
        model_checkpoint_file=e2e_args.checkpoint_name,
        args=e2e_args,
        device=device,
        src_lang=e2e_args.src_locale,
        tgt_lang=e2e_args.tgt_locale,
    )
    e2e_model.to(device)
    e2e_model.eval()
    logger.info(f'Arguments:\n{pformat(vars(e2e_args))}')
    logger.info(f'Loading from {e2e_args.best_checkpoint}')

    if args.nlg_type == 'neural':
        nlg_args = copy.deepcopy(args)
        nlg_args.path = args.nlg_path
        load_config_json(nlg_args)
        check_and_update_generation_args(nlg_args)
        NLGModel = getattr(models, nlg_args.model)
        nlg_model, _ = NLGModel.load(
            nlg_args.path,
            model_checkpoint_file=nlg_args.checkpoint_name,
            args=nlg_args,
            device=device,
            src_lang=nlg_args.src_locale,
            tgt_lang=nlg_args.tgt_locale,
        )
        nlg_model.to(device)
        nlg_model.eval()
        logger.info(f'Arguments:\n{pformat(vars(nlg_args))}')
        logger.info(f'Loading from {nlg_args.best_checkpoint}')
    else:
        _, filename = args.nlg_type.split('-')
        nlg_model = TemplateResponseGenerator(args.tgt_locale, filename=filename)

    return e2e_model, nlg_model


def main(args):
    e2e_model, nlg_model = init(args)
    loop = DialogueLoop(e2e_model, nlg_model)
    loop.run()
