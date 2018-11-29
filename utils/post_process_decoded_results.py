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


import sys
import os
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_sentences', default='./eval.en-tt.en', type=str)
parser.add_argument('--gold_program', default='./test/almond.gold.txt', type=str)
parser.add_argument('--predicted_program', default='./test/almond.txt', type=str)
parser.add_argument('--output_file', default='./test/out_file', type=str)

args = parser.parse_args()

def compute_accuracy(pred, gold):
    return pred == gold

def compute_grammar_accuracy(pred):
    return len(pred.split(' ')) != 0

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

with open(args.input_sentences, 'r') as input_file,\
        open(args.gold_program, 'r') as gold_file,\
        open(args.predicted_program, 'r') as pred_file,\
        open(args.output_file, 'w') as out:

    for line in zip(input_file, gold_file, pred_file):
        input, gold, pred = line
        input = input.replace(r'<s>', '')
        input = input.strip()
        gold = gold.strip()
        pred = pred.strip()
        accuracy = compute_accuracy(pred, gold)
        gramar_accuracy = compute_grammar_accuracy(pred)
        function_correctness = compute_funtion_correctness(pred, gold)
        correct_tokens = compute_correct_tokens(pred, gold)
        correct_quotes = compute_correct_quotes(pred, gold)

        out.write(input + ' || ' + gold + ' || ' + pred + ' || '
                  + str(accuracy) + ' || '
                  + str(gramar_accuracy) + '_grammar' + ' || '
                  + str(function_correctness) + '_function')
        if correct_quotes != False:
            out.write(' || ' + str("{0:.2f}".format(correct_quotes)) + '%_correct_quotes')
        if correct_tokens != False:
            out.write(' || ' + str("{0:.2f}".format(correct_tokens)) + '%_correct_tokens')
        out.write('\n')
