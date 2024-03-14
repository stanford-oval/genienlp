#!/usr/bin/env python3
# Copyright 2019 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Giovanni Campagna <gcampagn@cs.stanford.edu>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
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

import argparse

from . import (
    arguments,
    cache_embeddings,
    evaluate_file,
    export,
    predict,
    run_dialogue_loop,
    server,
    train,
)
from .paraphrase import run_generation
from .paraphrase.scripts import split_dataset
from .sts import sts_calculate_scores, sts_filter

subcommands = {
    # main commands
    'train': ('Train a model', arguments.parse_argv, train.main),
    'export': ('Export a trained model for serving', export.parse_argv, export.main),
    'predict': (
        'Evaluate a model by computing evaluation metrics on its predictions for a test dataset',
        predict.parse_argv,
        predict.main,
    ),
    'evaluate-file': (
        'Evaluate a file containing predictions and gold values by computing evaluation metrics for it',
        evaluate_file.parse_argv,
        evaluate_file.main,
    ),
    'server': ('Export RPC interface to predict', server.parse_argv, server.main),
    'cache-embeddings': ('Download and cache embeddings', cache_embeddings.parse_argv, cache_embeddings.main),
    'run-paraphrase': ('Run a paraphraser model', run_generation.parse_argv, run_generation.main),
    # commands that work with datasets
    'split-dataset': ('Split a dataset file into two files', split_dataset.parse_argv, split_dataset.main),
    # sts commands
    'sts-calculate-scores': (
        'Calculate semantic similarity scores between pairs of sentences',
        sts_calculate_scores.parse_argv,
        sts_calculate_scores.main,
    ),
    'sts-filter': (
        'Filter parallel sentences based on semantic similarity scores',
        sts_filter.parse_argv,
        sts_filter.main,
    ),
    # e2e dialogues
    'run-dialogue-loop': ('Interact with a dialogue agent', run_dialogue_loop.parse_argv, run_dialogue_loop.main),
}


def main():
    parser = argparse.ArgumentParser(prog='genienlp')
    subparsers = parser.add_subparsers(dest='subcommand')
    for subcommand in subcommands:
        helpstr, get_parser, command_fn = subcommands[subcommand]
        get_parser(subparsers.add_parser(subcommand, help=helpstr))

    argv = parser.parse_args()
    subcommands[argv.subcommand][2](argv)


if __name__ == '__main__':
    main()
