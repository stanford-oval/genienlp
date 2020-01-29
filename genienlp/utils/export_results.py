#!/usr/bin/python3
#
# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Mehrad Moradshahi <mehrad@cs.stanford.edu>
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
Created on Feb 25, 2019

@author: mehrad
'''


import sys
import os
import re
import argparse
from collections import defaultdict
from subprocess import Popen, PIPE
import csv

parser = argparse.ArgumentParser()

parser.add_argument('--log_file', default='./workdir/log_eval/log_eval', type=str)
parser.add_argument('--out_file', default='./workdir/log_eval/out.csv', type=str)

args = parser.parse_args()

METRICS = ['em', 'fm', 'dm', 'bleu', 'nf1']

def run():

    command = Popen(f'grep OrderedDict {args.log_file}'.split(' '), stdout=PIPE)
    result = command.communicate()[0]
    print(type(result))

    result = eval(result[len('OrderedDcit'):])

    results_dict = {k: f'{v:.2f}%' for (k,v) in result if k in METRICS}

    if not os.path.exists(args.out_file):
        with open(args.out_file, 'w+') as f_out:
            writer = csv.DictWriter(f_out, METRICS)
            writer.writeheader()

    with open(args.out_file, 'a') as f_out:
        writer = csv.DictWriter(f_out, METRICS)
        writer.writerow(results_dict)

if __name__ == '__main__':
    run()
