#!/usr/bin/python3
#
# Copyright 2017 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Mehrad Moradshahi <mehrad@cs.stanford.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
