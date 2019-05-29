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
Created on Aug 27, 2018

@author: mehrad
'''

from pprint import pprint
import sys
import os
import numpy as np
import re
import argparse
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()

parser.add_argument('--original_data', default='./test/cheatsheet.tsv', type=str, help='original dataset containing sentence ids')
parser.add_argument('--reference_gold', default='./test/test.en-tt.tt', type=str, help='gold thingtalk programs')
parser.add_argument('--gold_program', default='./test/almond.gold.txt', type=str, help='possibly shuffled gold thingtalk programs')
parser.add_argument('--predicted_program', default='./test/almond.txt', type=str, help='possibly shuffled predicted thingtalk programs')
parser.add_argument('--output_file', default='./test/out_file.tsv', type=str, help='output file to save results')
parser.add_argument('--is_shuffled', action='store_true', help='if prediction results are shuffled use find_indices to retrieve original ordering')

args = parser.parse_args()

def compute_accuracy(pred, gold):
    return pred == gold

def compute_accuracy_without_params(pred, gold):
    pred_list, gold_list = get_quotes(pred, gold)

    pred_cleaned = [pred.replace(val, '') for val in pred_list]
    gold_cleaned = [gold.replace(val, '') for val in gold_list]

    return pred_cleaned == gold_cleaned

def compute_grammar_accuracy(pred):
    return len(pred.split(' ')) != 0

def compute_device_correctness(pred, gold):
    return get_devices(pred) == get_devices(gold)

def get_devices(program):
    return [x.rsplit('.', 1)[0] for x in program.split(' ') if x.startswith('@')]

def compute_funtion_correctness(pred, gold):
    return get_functions(pred) == get_functions(gold)

def get_functions(program):
    return [x for x in program.split(' ') if x.startswith('@')]

def flatten(list):
    return [item for l in list for item in l]

def compute_correct_tokens(pred, gold):

    pred_list, gold_list = get_quotes(pred, gold)
    if len(gold_list) == 0:
        return False
    pred_list = flatten(map(lambda x: x.split(' '), pred_list))
    gold_list = flatten(map(lambda x: x.split(' '), gold_list))
    common = [token for token in gold_list if token in pred_list]
    return len(common) / len(gold_list) * 100.0

def compute_correct_quotes(pred, gold):

    pred_list, gold_list = get_quotes(pred, gold)
    if len(gold_list) == 0:
        return False
    common = [quote for quote in gold_list if quote in pred_list]
    return len(common) / len(gold_list) * 100.0


def get_quotes(pred, gold):

    quotes_list_pred = []
    quotes_list_gold = []
    quoted = re.compile('"[^"]*"')
    for value in quoted.findall(pred):
        quotes_list_pred.append(value)
    for value in quoted.findall(gold):
        quotes_list_gold.append(value)

    return quotes_list_pred, quotes_list_gold


def find_indices(ref, shuf):
    # models preprocess datasets before training and testing
    # during this procedure the dataset gets shuffled
    # this function finds a mapping between original ordering of data and shuffled data in the dataset
    ref_list = []
    shuf_list = []

    with open(ref, 'r') as f_ref:
        for line in f_ref:
            line = line[:-1].lower()
            ref_list.append(line)
    with open(shuf, 'r') as f_shuf:
        for line in f_shuf:
            line = line[1:-2].replace(r'\"', '"').replace(r'\/', '/').lower()
            shuf_list.append(line)

    indices = []
    for i, val in enumerate(shuf_list):
        found = ref_list.index(val)
        indices.append(found)

    if len(set(indices)) != len(indices):
        print('programs are not unique, thus index matching is not supported')
        print('list of not unique programs:')
        cntr = Counter(indices)
        pprint([ref_list[k] for k,v in cntr.items() if v > 1])
        sys.exit(1)

    return indices


def run():


    not_sorted_sents= []
    not_sorted_ids = []

    with open(args.original_data) as original_data:
        for line in original_data:
            id, sentence = line.strip().split('\t')[:2]
            not_sorted_ids.append(id)
            not_sorted_sents.append(sentence)
    if args.is_shuffled:
        indices = find_indices(args.reference_gold, args.gold_program)
        ids = [not_sorted_ids[i] for i in indices]
        sents = [not_sorted_sents[i] for i in indices]
    else:
        ids = not_sorted_ids
        sents= not_sorted_sents


    errors_dev = defaultdict(int)
    errors_func = defaultdict(lambda: defaultdict(int))

    cnt_dev = 0
    cnt_func = 0

    output_file_with_results = args.output_file
    output_file_raw = args.output_file.rsplit('.', 1)[0] + '_raw.' + args.output_file.rsplit('.', 1)[1]

    with open(args.gold_program, 'r') as gold_file,\
         open(args.predicted_program, 'r') as pred_file,\
         open(output_file_with_results, 'w') as out,\
         open(output_file_raw, 'w') as out_raw:

            for line in zip(sents, gold_file, pred_file, ids):
                input, gold, pred, id = line
                id = id.strip()
                input = input.replace(r'<s>', '').strip()
                gold = gold.strip().replace(r'\"', '"').replace(r'\/', '/')
                if gold[0] == gold[-1] == '"':
                    gold = gold[1:-1]
                pred = pred.strip().replace(r'\"', '"').replace(r'\/', '/')
                if pred[0] == pred[-1] == '"':
                    pred = pred[1:-1]

                out_raw.write(id + '\t' + input + '\t' + gold + '\t' + pred)
                out_raw.write('\n')


                accuracy = compute_accuracy(pred, gold)
                accuracy_without_params = compute_accuracy_without_params(pred, gold)
                grammar_accuracy = compute_grammar_accuracy(pred)
                function_correctness = compute_funtion_correctness(pred, gold)
                device_correctness = compute_device_correctness(pred, gold)
                correct_tokens = compute_correct_tokens(pred, gold)
                correct_quotes = compute_correct_quotes(pred, gold)

                ##########
                # error analysis
                if not device_correctness:
                    gold_devs = get_devices(gold)
                    pred_devs = get_devices(pred)
                    cnt_dev += 1
                    if len(gold_devs) == len(pred_devs):

                        for i, g in enumerate(gold_devs):
                            if g != pred_devs[i]:
                                errors_dev[(g, pred_devs[i])] += 1

                elif not function_correctness:
                    gold_funcs = get_functions(gold)
                    pred_funcs = get_functions(pred)
                    cnt_func += 1
                    if len(gold_funcs) == len(pred_funcs):
                        devices = get_devices(gold)
                        for i, d in enumerate(devices):
                            if gold_funcs[i] != pred_funcs[i]:
                                errors_func[d][(gold_funcs[i].rsplit('.', 1)[1], pred_funcs[i].rsplit('.', 1)[1])] += 1
                ##########

                out.write(input + ' || ' + gold + ' || ' + pred + ' || '
                          + str(accuracy) + ' || '
                          + str(accuracy_without_params) + '_w/o_params' + ' || '
                          + str(grammar_accuracy) + '_grammar' + ' || '
                          + str(function_correctness) + '_function' + ' || '
                          + str(device_correctness) + '_device')
                if correct_quotes != False:
                    out.write(' || ' + str("{0:.2f}".format(correct_quotes)) + '%_correct_quotes')
                if correct_tokens != False:
                    out.write(' || ' + str("{0:.2f}".format(correct_tokens)) + '%_correct_tokens')
                out.write('\n')
                out.write('\n')
                out.write('\n')

    print('cnt_dev: ', cnt_dev)
    print('cnt_func: ', cnt_func)
    print('errors_dev: ', errors_dev.items())
    print('errors_func: ', errors_func.items())

if __name__ == '__main__':
    run()