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

import sys
import argparse

from . import arguments, train, predict, server, cache_embeddings

subcommands = {
    'train': ('Train a model', arguments.parse_argv, train.main),
    'predict': ('Evaluate a model, or compute predictions on a test dataset', predict.parse_argv, predict.main),
    'server': ('Export RPC interface to predict', server.parse_argv, server.main),
    'cache-embeddings': ('Download and cache embeddings', cache_embeddings.parse_argv, cache_embeddings.main),
}


def usage():
    print('Usage: %s SUBCOMMAND [OPTIONS]' % (sys.argv[0]), file=sys.stderr)
    print(file=sys.stderr)
    print('Available subcommands:', file=sys.stderr)
    for subcommand, (help_text, _) in subcommands.items():
        print('  %s - %s' % (subcommand, help_text), file=sys.stderr)
    sys.exit(1)


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
