# Copyright 2021 The Board of Trustees of the Leland Stanford Junior University
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


import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def parse_argv(parser):
    parser.add_argument('--eval_results', nargs='+', required=True, help='path to eval json files')


def main(args):

    # write metrics
    result = defaultdict(list)

    num_files = len(args.eval_results)
    if num_files > 2:
        logger.warning('only metrics for first two splits will be shown')
    max_size = 2 if num_files == 1 else 1

    for file in args.eval_results:
        with open(file, 'r') as fin:
            metrics = json.load(fin)

        # get split name
        split = file.rsplit('/', 2)[-2]

        # kf dashboard is hardcoded to only show two metrics
        # we show the first metric (i.e. decaScore) for each split if num_files is 1, otherwise the first two metrics for provided split

        extracted_metrics = [
            {
                'name': split + '-' + key,
                'numberValue': val / 100,
                'format': "PERCENTAGE",
            }
            for key, val in list(metrics.items())[:max_size]
        ]

        result['metrics'].extend(extracted_metrics)

    with open('/tmp/mlpipeline-metrics.json', 'w') as fout:
        json.dump(result, fout)

    # write UI metadata
    metdata = defaultdict(list)

    for file in args.eval_results:
        with open(file, 'r') as fin:
            metrics = json.load(fin)

        # get split name
        split = file.rsplit('/', 2)[-2]

        extracted_metadata = [
            {
                'storage': 'inline',
                'source': ','.join([split] + list(map(str, metrics.values()))),
                "format": 'csv',
                "type": "table",
                "header": ["eval_set"] + list(metrics.keys()),
            }
        ]

        metdata['outputs'].extend(extracted_metadata)

    with open('/tmp/mlpipeline-ui-metadata.json', 'w') as fout:
        json.dump(metdata, fout)
