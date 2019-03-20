#
# Copyright (c) 2017-2019, The Board of Trustees of the Leland Stanford Junior University
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
from collections import OrderedDict

from . import slr
from .slr import generator as slr_generator


class ShiftReduceGrammar:

    def __init__(self, logger=None):
        self.tokens = ['<pad>', '</s>', '<s>']
        
        self._logger = logger
        self._parser = None

    @property
    def num_control_tokens(self):
        return 3

    def construct_parser(self, grammar):

        generator = slr_generator.SLRParserGenerator(grammar, '$input')
        self._parser = generator.build()
        
        if self._logger:
            self._logger.info('num rules', self._parser.num_rules)
            self._logger.info('num states', self._parser.num_states)
            self._logger.info('num shifts', 1)
            self._logger.info('num terminals', len(generator.terminals))

        self.dictionary = generator.dictionary
        self._word_id = self.dictionary['WORD']
        self.tokens = generator.terminals
    
    def tokenize_program(self, program):
        if isinstance(program, str):
            program = program.split(' ')
        for token in program:
            yield self.dictionary[token], None

    def preprocess_program(self, program,
                           direction='bottomup',
                           max_length=None):
        assert direction in ('bottomup', 'topdown')

        tokenizer = self.tokenize_program(program)

        if direction == 'topdown':
            parsed = self._parser.parse_reverse(tokenizer)
        else:
            parsed = self._parser.parse(tokenizer)

        if max_length is None:
            # conservative estimate: the actual length will be smaller
            # because we don't need as many shifts
            max_length = len(parsed) + 1

        output = []
        for action, param in parsed:
            assert action in (slr.SHIFT_CODE, slr.REDUCE_CODE)
            if action == slr.SHIFT_CODE:
                term_id, payload = param
                term = self.tokens[term_id]
                if term_id == self._word_id:
                    assert not payload.startswith('R')
                    output.append(payload)
                else:
                    continue
            else:
                output.append('R' + str(param))

        return output

    def reconstruct_program(self, sequence, direction='bottomup', ignore_errors=False):
        def gen_action():
            for x in sequence:
                if x.startswith('R'):
                    yield (slr.REDUCE_CODE, int(x[1:]))
                else:
                    yield (slr.SHIFT_CODE, (self._word_id, x))
            yield (slr.ACCEPT_CODE, None)

        try:
            if direction == 'topdown':
                term_ids = self._parser.reconstruct_reverse(gen_action())
            else:
                term_ids = self._parser.reconstruct(gen_action())
        except (KeyError, IndexError, ValueError):
            if ignore_errors:
                # the NN generated something that does not conform to the grammar,
                # ignore it
                return
            else:
                raise

        for i, (term_id, payload) in enumerate(term_ids):
            if payload is not None:
                yield payload
            else:
                yield self.tokens[term_id]
        
    def print_all_actions(self):
        print(0, 'P', 'pad')
        print(1, 'A', 'accept')
        print(2, 'G', 'start')
        for i, (lhs, rhs) in enumerate(self._parser.rules):
            print(i+self.num_control_tokens, 'R' + str(i), 'reduce', lhs, '->', ' '.join(rhs))
        print(self._word_id, 'S0', 'copy', 'WORD')

    def _action_to_print_full(self, action):
        if action == slr.PAD_ID:
            return ('pad',)
        elif action == slr.EOF_ID:
            return ('accept',)
        elif action == slr.START_ID:
            return ('start',)
        elif action - self.num_control_tokens < self._parser.num_rules:
            lhs, rhs = self._parser.rules[action - self.num_control_tokens]
            return ('reduce', ':', lhs, '->', ' '.join(rhs))
        else:
            return ('shift', 'WORD')

    def print_prediction(self, input_sentence, sequences):
        actions = sequences['actions']
        for i, action in enumerate(actions):
            if action == slr.PAD_ID:
                print(action, 'pad')
                break
            elif action == slr.EOF_ID:
                print(action, 'accept') 
                break
            elif action == slr.START_ID:
                print(action, 'start')
            elif action - self.num_control_tokens < self._parser.num_rules:
                lhs, rhs = self._parser.rules[action - self.num_control_tokens]
                print(action, 'reduce', ':', lhs, '->', ' '.join(rhs))
            else:
                term = 'WORD'
                print(action, 'shift', term, sequences[term][i], self._parser.extensible_terminals[term][sequences[term][i]])
    
    def prediction_to_string(self, sequences):
        def action_to_string(action):
            if action == slr.PAD_ID:
                return 'P'
            elif action == slr.EOF_ID:
                return 'A'
            elif action == slr.START_ID:
                return 'G'
            elif action - self.num_control_tokens < self._parser.num_rules:
                return 'R' + str(action - self.num_control_tokens)
            else:
                return 'S' + str(action - self.num_control_tokens - self._parser.num_rules)
        return list(map(action_to_string, sequences['actions']))

    def string_to_prediction(self, strings):
        def string_to_action(string):
            if string == 'P':
                return slr.PAD_ID
            elif string == 'A':
                return slr.EOF_ID
            elif string == 'G':
                return slr.START_ID
            elif string.startswith('R'):
                action = int(string[1:]) + self.num_control_tokens
                assert action - self.num_control_tokens < self._parser.num_rules
                return action
            else:
                action = int(string[1:]) + self.num_control_tokens + self._parser.num_rules
                return action
        return list(map(string_to_action, strings))
