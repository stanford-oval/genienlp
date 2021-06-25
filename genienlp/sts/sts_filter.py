#
# Copyright (c) 2020-2021 The Board of Trustees of the Leland Stanford Junior University
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


import numpy as np


def parse_argv(parser):

    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--filtering_metric', type=str, default='constant', choices=['mean', 'mean+std', 'all', 'constant'])
    parser.add_argument(
        '--filtering_threshold',
        type=float,
        help='STS threshold score used to filter sentences if filtering_metric is constant',
    )


def main(args):

    all_scores = []

    with open(args.input_file, 'r') as fin:
        for line in fin:
            parts = list(map(lambda p: p.strip(), line.split('\t')))
            id_, orig_sent, para_sent, score, program = parts
            all_scores.append(score)

    all_scores = np.array(all_scores, dtype=float)
    scores_mean = np.mean(all_scores)
    scoers_std = np.std(all_scores)

    if args.filtering_metric == 'mean':
        accepted_ids = all_scores >= scores_mean
    elif args.filtering_metric == 'mean+std':
        accepted_ids = all_scores >= (scores_mean + scoers_std)
    elif args.filtering_metric == 'constant':
        assert args.filtering_threshold is not None
        accepted_ids = all_scores >= args.filtering_threshold
    # accept all
    else:
        accepted_ids = all_scores >= 0.0

    with open(args.input_file, 'r') as fin, open(args.output_file, 'w') as fout:
        for i, line in enumerate(fin):
            if accepted_ids[i]:
                parts = list(map(lambda p: p.strip(), line.split('\t')))
                id_, orig_sent, para_sent, score, program = parts
                fout.write('\t'.join([id_, para_sent, program]) + '\n')
