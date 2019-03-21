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


import itertools
import sys
import numpy as np

from ..slr import EOF_TOKEN, PAD_TOKEN, START_TOKEN, \
    PAD_ID, EOF_ID, START_ID, \
    ACCEPT_CODE, SHIFT_CODE, REDUCE_CODE, INVALID_CODE
from .np_parser import ShiftReduceParser


class ItemSetInfo:
    def __init__(self):
        self.id = 0
        self.intransitions = set()
        self.outtransitions = set()


class ItemSet:
    def __init__(self, rules):
        self.rules = list(rules)
    def __hash__(self):
        h = 0
        for el in self.rules:
            h ^= hash(el)
        return h
    def __eq__(self, other):
        return self.rules == other.rules

DEBUG = False

ITEM_SET_SEP = ''


class SLRParserGenerator():
    '''
    Construct a shift-reduce parser given an SLR grammar.
    
    The grammar must be binarized beforehand.
    '''
    
    def __init__(self, grammar, start_symbol):
        # optimizations first
        self._start_symbol = start_symbol
        self._optimize_grammar(grammar)
        grammar['$ROOT'] = [(start_symbol, EOF_TOKEN)]
        
        self._number_rules(grammar)
        self._extract_terminals_non_terminals()
        self._build_first_sets()
        self._build_follow_sets()
        self._generate_all_item_sets()
        self._build_state_transition_matrix()
        self._build_parse_tables()
        
        self._check_first_sets()
        self._check_follow_sets()
        
    def build(self):
        # the last rule is $ROOT -> $input <<EOF>>
        # which is a pseudo-rule needed for the SLR generator
        # we ignore it here
        return ShiftReduceParser(self.rules[:-1], self.rule_table, self.action_table, self.goto_table,
                                 self.terminals, self._all_dictionary,
                                 self._all_dictionary[self._start_symbol])
    
    def _optimize_grammar(self, grammar):
        progress = True
        i = 0
        while progress:
            progress = False
            if DEBUG:
                print("Optimization pass", i+1)            
            progress = self._remove_empty_nonterminals(grammar) or progress
            progress = self._remove_unreachable_nonterminals(grammar) or progress
    
    def _remove_empty_nonterminals(self, grammar):
        progress = True
        any_change = False
        deleted = set()
        while progress:
            progress = False
            for lhs, rules in grammar.items():
                if len(rules) == 0:
                    if lhs not in deleted:
                        if DEBUG:
                            print("Non-terminal", lhs, "is empty, deleted")
                        progress = True
                        any_change = True
                    deleted.add(lhs)
                else:
                    new_rules = []
                    any_rules_deleted = False
                    for rule in rules:
                        rule_is_deleted = False
                        for rhs in rule:
                            if rhs in deleted:
                                rule_is_deleted = True
                                break
                        if not rule_is_deleted:
                            new_rules.append(rule)
                        else:
                            if DEBUG:
                                print("Rule", lhs, "->", rule, "deleted")
                            any_rules_deleted = True
                    if any_rules_deleted:
                        grammar[lhs] = new_rules
                        progress = True
                        any_change = True
        for lhs in deleted:
            del grammar[lhs]
        return any_change
    
    def _remove_unreachable_nonterminals(self, grammar):
        stack = [self._start_symbol]
        visited = set()
        while len(stack) > 0:
            nonterm = stack.pop()
            if nonterm in visited:
                continue
            visited.add(nonterm)
            for rhs in grammar[nonterm]:
                for rhs_token in rhs:
                    if rhs_token[0] == '$' and rhs_token not in visited:
                        stack.append(rhs_token)
                        if not rhs_token in grammar:
                            raise ValueError("Non-terminal " + str(rhs_token) + " does not exist, in rule " + nonterm + " -> " + str(rhs))
                        
        todelete = set()
        anychange = False
        for lhs in grammar:
            if lhs not in visited:
                if DEBUG:
                    print("Non-terminal " + lhs + " is not reachable, deleted")
                todelete.add(lhs)
                anychange = True
        for lhs in todelete:
            del grammar[lhs]
        return anychange
    
    def _check_first_sets(self):
        for lhs, first_set in self._first_sets.items():
            if len(first_set) == 0:
                print("WARNING: non-terminal " + lhs + " cannot start with any terminal")
    
    def _check_follow_sets(self):
        for lhs, follow_set in self._follow_sets.items():
            if lhs == '$ROOT':
                continue
            if len(follow_set) == 0:
                print("WARNING: non-terminal " + lhs + " cannot be followed by any terminal")
    
    def _extract_terminals_non_terminals(self):
        terminals = { PAD_TOKEN, EOF_TOKEN, START_TOKEN }
        non_terminals = set()
        for lhs, rule in self.rules:
            non_terminals.add(lhs)
            for rhs in rule:
                assert isinstance(rhs, str)
                if rhs[0] != '$':
                    terminals.add(rhs)
                else:
                    non_terminals.add(rhs)
        
        self.terminals = list(terminals)
        self.terminals.sort()
        assert self.terminals[PAD_ID] == PAD_TOKEN
        assert self.terminals[EOF_ID] == EOF_TOKEN
        assert self.terminals[START_ID] == START_TOKEN
        self.non_terminals = list(non_terminals)
        self.non_terminals.sort()
        self.dictionary = dict((token, i) for i, token in enumerate(self.terminals))
        self._all_dictionary = dict((token, i) for i, token in
                                    enumerate(itertools.chain(self.terminals, self.non_terminals)))

    def print_rules(self, fp=sys.stdout):
        for i, (lhs, rhs) in enumerate(self.rules):
            print(i, lhs, '->', ' '.join(rhs), file=fp)

    def _number_rules(self, grammar):
        self.rules = []
        self.grammar = dict()
        self.rule_set = set()
        for lhs, rules in grammar.items():
            self.grammar[lhs] = []
            for rule in rules:
                if not isinstance(rule, tuple):
                    raise TypeError('Invalid rule ' + repr(rule))
                #assert len(rule) == 1 or len(rule) == 2
                rule_id = len(self.rules)
                self.rules.append((lhs, rule))
                if (lhs, rule) in self.rule_set:
                    raise ValueError('duplicate rule ' + lhs + '->' + str(rule))
                self.grammar[lhs].append(rule_id)
                if DEBUG:
                    print(rule_id, lhs, '->', rule)

    def _item_set_followers(self, item_set):
        for rule in item_set.rules:
            _, rhs = rule
            for i in range(len(rhs)-1):
                if rhs[i] == ITEM_SET_SEP and rhs[i+1] != EOF_TOKEN:
                    yield rhs[i+1]

    def _advance(self, item_set, token):
        for rule in item_set.rules:
            rule_id, rhs = rule
            for i in range(len(rhs)-1):
                if rhs[i] == ITEM_SET_SEP and rhs[i+1] == token:
                    yield rule_id, (rhs[:i] + (token, ITEM_SET_SEP) + rhs[i+2:])
                    break
        
    def _make_item_set(self, lhs):
        for rule_id in self.grammar[lhs]:
            lhs, rhs = self.rules[rule_id]
            yield rule_id, (ITEM_SET_SEP,) + rhs
    
    def _close(self, items):
        def _is_nonterminal(symbol):
            return symbol[0] == '$'
        
        item_set = set(items)
        stack = list(item_set)
        while len(stack) > 0:
            item = stack.pop()
            _, rhs = item
            for i in range(len(rhs)-1):
                if rhs[i] == ITEM_SET_SEP and _is_nonterminal(rhs[i+1]):
                    for new_rule in self._make_item_set(rhs[i+1]):
                        if new_rule in item_set:
                            continue
                        item_set.add(new_rule)
                        stack.append(new_rule)
                    break
        item_set = list(item_set)
        item_set.sort()
        return item_set
    
    def _generate_all_item_sets(self):
        item_sets = dict()
        i = 0
        item_set0 = ItemSet(self._close(self._make_item_set('$ROOT')))
        item_set0_info = ItemSetInfo()
        item_sets[item_set0] = item_set0_info
        i += 1
        queue = []
        queue.append(item_set0)
        while len(queue) > 0:
            item_set = queue.pop(0)
            my_info = item_sets[item_set]
            for next_token in self._item_set_followers(item_set):
                new_set = ItemSet(self._close(self._advance(item_set, next_token)))
                if new_set in item_sets:
                    info = item_sets[new_set]
                else:
                    info = ItemSetInfo()
                    info.id = i
                    i += 1
                    item_sets[new_set] = info
                    queue.append(new_set)
                info.intransitions.add((my_info.id, next_token))
                my_info.outtransitions.add((info.id, next_token))
        
        for item_set, info in item_sets.items():
            item_set.info = info
            if DEBUG:
                print("Item Set", item_set.info.id, item_set.info.intransitions)
                for rule in item_set.rules:
                    rule_id, rhs = rule
                    lhs, _ = self.rules[rule_id]
                    print(rule_id, lhs, '->', rhs)
                print()
            
        item_set_list = [None] * len(item_sets.keys())
        for item_set in item_sets.keys():
            item_set_list[item_set.info.id] = item_set
        self._item_sets = item_set_list
        self._n_states = len(self._item_sets)
    
    def _build_state_transition_matrix(self):
        self._state_transition_matrix = [dict() for  _ in range(self._n_states)]
        
        for item_set in self._item_sets:
            for next_id, next_token in item_set.info.outtransitions:
                if next_token in self._state_transition_matrix[item_set.info.id]:
                    raise ValueError("Ambiguous transition from", item_set.info.id, "through", next_token, "to", self._state_transition_matrix[item_set.info.id], "and", next_id)
                self._state_transition_matrix[item_set.info.id][next_token] = next_id
                
    def _build_first_sets(self):
        def _is_terminal(symbol):
            return symbol[0] != '$'
        
        first_sets = dict()
        for nonterm in self.non_terminals:
            first_sets[nonterm] = set()
        progress = True
        while progress:
            progress = False
            for lhs, rules in self.grammar.items():
                union = set()
                for rule_id in rules:
                    _, rule = self.rules[rule_id]
                    # Note: our grammar doesn't include rules of the form A -> epsilon
                    # because it's meant for an SLR parser not an LL parser, so this is
                    # simpler than what Wikipedia describes in the LL parser article
                    if _is_terminal(rule[0]):
                        first_set_rule = set([rule[0]])
                    else:
                        first_set_rule = first_sets.get(rule[0], set())
                    union |= first_set_rule
                if union != first_sets[lhs]:
                    first_sets[lhs] = union
                    progress = True
                    
        self._first_sets = first_sets
        
    def _build_follow_sets(self):
        follow_sets = dict()
        for nonterm in self.non_terminals:
            follow_sets[nonterm] = set()
        
        progress = True
        def _add_all(from_set, into_set):
            progress = False
            for v in from_set:
                if v not in into_set:
                    into_set.add(v)
                    progress = True
            return progress
        def _is_nonterminal(symbol):
            return symbol[0] == '$'
        
        while progress:
            progress = False
            for lhs, rule in self.rules:
                for i in range(len(rule)-1):
                    if _is_nonterminal(rule[i]):
                        if _is_nonterminal(rule[i+1]):
                            #if 'not' in self._first_sets[rule[i+1]]:
                            #    print(rule[i], 'followed by', rule[i+1])
                            progress = _add_all(self._first_sets[rule[i+1]], follow_sets[rule[i]]) or progress
                        else:
                            if rule[i+1] not in follow_sets[rule[i]]:
                                follow_sets[rule[i]].add(rule[i+1])
                                progress = True
                if _is_nonterminal(rule[-1]):
                    #if 'not' in follow_sets[lhs]:
                    #    print(lhs, 'into', rule[-1])
                    progress = _add_all(follow_sets[lhs], follow_sets[rule[-1]]) or progress
                                    
        self._follow_sets = follow_sets
        if DEBUG:
            print()
            print("Follow sets")
            for nonterm, follow_set in follow_sets.items():
                print(nonterm, "->", follow_set)
            
    def _build_parse_tables(self):
        self.goto_table = np.full(fill_value=INVALID_CODE,
                                  shape=(self._n_states, len(self.non_terminals)),
                                  dtype=np.int32)
        self.action_table = np.full(fill_value=INVALID_CODE,
                                    shape=(self._n_states, len(self.terminals) + len(self.non_terminals), 2),
                                    dtype=np.int32)
        
        self.rule_table = np.empty(shape=(len(self.rules),2), dtype=np.int32)
        for rule_id, (lhs, rhs) in enumerate(self.rules):
            self.rule_table[rule_id, 0] = self._all_dictionary[lhs] - len(self.terminals)
            self.rule_table[rule_id, 1] = len(rhs)
        
        for nonterm_id, nonterm in enumerate(self.non_terminals):
            for i in range(self._n_states):
                if nonterm in self._state_transition_matrix[i]:
                    self.goto_table[i, nonterm_id] = self._state_transition_matrix[i][nonterm]
        for term in self.terminals:
            for i in range(self._n_states):
                if term in self._state_transition_matrix[i]:
                    term_id = self.dictionary[term]
                    self.action_table[i, term_id, 0] = SHIFT_CODE
                    self.action_table[i, term_id, 1] = self._state_transition_matrix[i][term]
                    
        for item_set in self._item_sets:
            for item in item_set.rules:
                _, rhs = item
                for i in range(len(rhs)-1):
                    if rhs[i] == ITEM_SET_SEP and rhs[i+1] == EOF_TOKEN:
                        self.action_table[item_set.info.id, EOF_ID, 0] = ACCEPT_CODE
                        self.action_table[item_set.info.id, EOF_ID, 1] = INVALID_CODE
        
        for item_set in self._item_sets:
            for item in item_set.rules:
                rule_id, rhs = item
                if rhs[-1] != ITEM_SET_SEP:
                    continue
                lhs, _ = self.rules[rule_id]
                for term_id, term in enumerate(self.terminals):
                    if term in self._follow_sets.get(lhs, set()):
                        if self.action_table[item_set.info.id, term_id, 0] != INVALID_CODE \
                         and not (self.action_table[item_set.info.id, term_id, 0] == REDUCE_CODE and
                                  self.action_table[item_set.info.id, term_id, 1] == rule_id):
                            print("Item Set", item_set.info.id, item_set.info.intransitions)
                            for rule in item_set.rules:
                                loop_rule_id, rhs = rule
                                lhs, _ = self.rules[loop_rule_id]
                                print(loop_rule_id, lhs, '->', rhs)
                            print()
                            #if self.action_table[item_set.info.id][term][0] == 'shift':
                            #    jump_set = self._item_sets[self.action_table[item_set.info.id][term][1]]
                            #    print("Item Set", jump_set.info.id, jump_set.info.intransitions)
                            #    for rule in jump_set.rules:
                            #        loop_rule_id, rhs = rule
                            #        lhs, _ = self.rules[loop_rule_id]
                            #        print(loop_rule_id, lhs, '->', rhs)
                            #    print()
                            
                            raise ValueError("Conflict for state", item_set.info.id, "terminal", term, "want", ("reduce", rule_id), "have", self.action_table[item_set.info.id, term_id])
                        self.action_table[item_set.info.id, term_id, 0] = REDUCE_CODE
                        self.action_table[item_set.info.id, term_id, 1] = rule_id

