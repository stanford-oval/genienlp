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


from collections import defaultdict

from ..slr import EOF_ID, ACCEPT_CODE, REDUCE_CODE, SHIFT_CODE, INVALID_CODE


class ShiftReduceParser:
    '''
    A bottom-up parser for a deterministic CFG language, based on shift-reduce
    tables.
    
    The parser can transform a string in the language to a sequence of
    shifts and reduces, and can transform a valid sequence of reduces to
    a string in the language.
    '''
    
    def __init__(self, rules, rule_table, action_table, goto_table, terminals, dictionary, start_symbol):
        super().__init__()
        self.rules = rules
        self.rule_table = rule_table
        self._action_table = action_table
        self._goto_table = goto_table
        self.terminals = terminals
        self.dictionary = dictionary
        self._start_symbol = start_symbol

    @property
    def num_rules(self):
        # the last rule is $ROOT -> $input <<EOF>>
        # which is a pseudo-rule needed for the SLR generator
        # we ignore it here
        return len(self.rules)
    
    @property
    def num_states(self):
        return len(self._action_table)
    
    def parse_reverse(self, sequence):
        bottom_up_sequence = self.parse(sequence)
        lens = [None] * len(bottom_up_sequence)
        children = [None] * len(bottom_up_sequence)
        tree = [None] * len(bottom_up_sequence)
        i = 0
        for action, param in bottom_up_sequence:
            my_length = 1
            my_children = []
            if action == REDUCE_CODE:
                _, rhssize = self.rule_table[param]
                current_child = i-1
                for _ in range(rhssize):
                    my_children.append(current_child)
                    my_length += lens[current_child]
                    current_child -= lens[current_child]
            lens[i] = my_length
            tree[i] = (action,param)
            children[i] = tuple(reversed(my_children))
            i += 1
        reversed_sequence = []
        def write_subsequence(node, start):
            reversed_sequence.append(tree[node])
            for c in children[node]:
                write_subsequence(c, start)
                start += lens[c]
        write_subsequence(i-1, 0)
        return reversed_sequence

    def parse(self, sequence):
        stack = [0]
        state = 0
        result = []
        sequence_iter = iter(sequence)
        terminal_id, token = next(sequence_iter)
        while True:
            if self._action_table[state, terminal_id, 0] == INVALID_CODE:
                expected_token_ids,  = self._action_table[state, :, 0].nonzero()
                expected_tokens = [self.terminals[i] for i in expected_token_ids]
                
                raise ValueError(
                    "Parse error: unexpected token " + self.terminals[terminal_id] + " in state " + str(state) + ", expected " + str(
                        expected_tokens))
            action, param = self._action_table[state, terminal_id]
            if action == ACCEPT_CODE:
                return result
            #if action == 'shift':
            #    print('shift', param, token)
            #else:
            #    print('reduce', param, self.rules[param])
            if action == SHIFT_CODE:
                state = param
                result.append((SHIFT_CODE, (terminal_id, token)))
                stack.append(state)
                try:
                    terminal_id, token = next(sequence_iter)
                except StopIteration:
                    terminal_id = EOF_ID
            else:
                assert action == REDUCE_CODE
                rule_id = param
                result.append((REDUCE_CODE, rule_id))
                lhs_id, rhssize = self.rule_table[rule_id]
                for _ in range(rhssize):
                    stack.pop()
                state = stack[-1]
                state = self._goto_table[state, lhs_id]
                stack.append(state)
                
    def reconstruct_reverse(self, sequence):
        output_sequence = []
        if not isinstance(sequence, list):
            sequence = list(sequence)
        
        # all the trickyness in this method comes
        # from the fact that all unnecessary shifts
        # can be omitted and we should still be able
        # to reconstruct
        # (this is important because letting the RNN
        # generate unnecessary shifts is wasteful of
        # model capacity)
        # at the same time, whether a shift is necessary
        # or not is decided at a higher level, so we
        # try to accomodate all correct sequences here
        # (and do something weird for incorrect sequences)
    
        def recurse(start_at):
            action, param = sequence[start_at]
            if action != REDUCE_CODE:
                raise ValueError('Invalid action, expected reduce')
            _, rhs = self.rules[param]
            length = 1
            for symbol in rhs:
                if symbol.startswith('$'):
                    length += recurse(start_at + length)
                else:
                    symbol_id = self.dictionary[symbol]
                    # check if we have a shift element as child
                    # if so, we consume it and output it,
                    # otherwise we output just symbol_id
                    if start_at + length < len(sequence) and \
                        sequence[start_at + length][0] == SHIFT_CODE:
                        _, (token_id, payload) = sequence[start_at + length]
                        if symbol_id != token_id:
                            # this could happen if the rule has two
                            # terminals in a row, and the shift for
                            # the first one was elided by the second one
                            # was not
                            # in this case, we emit symbol_id as if there
                            # was no shift here
                            
                            # NOTE: we rely on token_ids being elided consistently
                            # ie, either all instances of a certain token_id are
                            # omitted or they are present
                            output_sequence.append((symbol_id, None))
                            continue
                        
                        # append the token to the output and consume
                        # the shift
                        output_sequence.append((token_id, payload))
                        length += 1
                    else:
                        # the shift was elided, consume nothing and output
                        # just the symbol id
                        output_sequence.append((symbol_id, None))
            return length
    
        recurse(0)
        return output_sequence

    def reconstruct(self, sequence):
        # beware: this code is tricky
        # don't approach it without patience, pen and paper and
        # many test cases

        stack = []
        top_stack_id = None
        token_stacks = defaultdict(list)
        for action, param in sequence:
            if action == ACCEPT_CODE:
                break
            elif action == SHIFT_CODE:
                term_id, token = param
                token_stacks[term_id].append(token)
            else:
                assert action == REDUCE_CODE
                rule_id = param
                lhs, rhs = self.rules[rule_id]
                top_stack_id = self.rule_table[rule_id, 0]
                if len(rhs) == 1:
                    #print("fast path for", lhs, "->", rhs)
                    # fast path for unary rules
                    symbol = rhs[0]
                    if symbol[0] == '$':
                        # unary non-term to non-term, no stack manipulation
                        continue
                    # unary term to non-term, we push directly to the stack
                    # a list containing a single item, the terminal and its data
                    symbol_id = self.dictionary[symbol]
                    if symbol_id in token_stacks and token_stacks[symbol_id]:
                        token_stack = token_stacks[symbol_id]
                        stack.append([(symbol_id, token_stack.pop())])
                        if not token_stack:
                            del token_stacks[symbol_id]
                    else:
                        stack.append([(symbol_id, None)])
                    #print(stack)
                else:
                    #print("slow path for", lhs, "->", rhs)
                    new_prog = []
                    for symbol in reversed(rhs):
                        if symbol[0] == '$':
                            new_prog.extend(stack.pop())
                        else:
                            symbol_id = self.dictionary[symbol]
                            if symbol_id in token_stacks and token_stacks[symbol_id]:
                                token_stack = token_stacks[symbol_id]
                                new_prog.append((symbol_id, token_stack.pop()))
                                if not token_stack:
                                    del token_stacks[symbol_id]
                            else:
                                new_prog.append((symbol_id, None))
                    stack.append(new_prog)
                    #print(stack)
        #print("Stack", stack)
        if top_stack_id is None or \
            len(self.terminals) + top_stack_id != self._start_symbol or \
            len(stack) != 1:
            raise ValueError("Invalid sequence")
        
        assert len(stack) == 1
        # the program is constructed on the stack in reverse order
        # bring it back to the right order
        stack[0].reverse()
        return stack[0]


