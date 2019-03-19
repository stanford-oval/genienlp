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

import json
import sys

from collections import OrderedDict
from orderedset import OrderedSet

from .shift_reduce_grammar import ShiftReduceGrammar


SPECIAL_TOKENS = ['special:yes', 'special:no', 'special:nevermind',
                  'special:makerule', 'special:failed', 'special:help',
                  'special:thank_you', 'special:hello',
                  'special:sorry', 'special:cool', 'special:wakeup']
TYPES = {
    'Location': (['=='], ['LOCATION', 'location:current_location', 'location:work', 'location:home']),
    'Boolean':  (['=='], ['true', 'false']),
    'String': (['==', '=~', '~=', 'starts_with', 'ends_with'], ['""', ('"', '$word_list', '"'), 'QUOTED_STRING']),
    'Date': (['==', '>=', '<='], [
        'DATE',
        'now',
        ('start_of', 'unit:h'),
        ('start_of', 'unit:day'),
        ('start_of', 'unit:week'),
        ('start_of', 'unit:mon'),
        ('start_of', 'unit:year'),
        ('end_of', 'unit:h'),
        ('end_of', 'unit:day'),
        ('end_of', 'unit:week'),
        ('end_of', 'unit:mon'),
        ('end_of', 'unit:year'),
        ('$constant_Date', '+', '$constant_Measure(ms)'),
        ('$constant_Date', '-', '$constant_Measure(ms)'),
        ('$constant_Time',),
        ]),
    'Time': (['=='], ['TIME']),
    'Currency': (['==', '>=', '<='], ['CURRENCY']),
    'Number': (['==', '>=', '<='], ['NUMBER', '1', '0']),
    'Entity(tt:username)': (['=='], ['USERNAME', ('"', '$word_list', '"', '^^tt:username')]),
    'Entity(tt:hashtag)': (['=='], ['HASHTAG', ('"', '$word_list', '"', '^^tt:hashtag')]),
    'Entity(tt:phone_number)': (['=='], ['PHONE_NUMBER']),
    'Entity(tt:email_address)': (['=='], ['EMAIL_ADDRESS']),
    'Entity(tt:url)': (['=='], ['URL']),
    'Entity(tt:path_name)': (['=='], ['PATH_NAME']),
    'Entity(tt:picture)': (['=='], []),
    'Entity(tt:program)': (['=='], [])
}
TYPE_RENAMES = {
    'Username': 'Entity(tt:username)',
    'Hashtag': 'Entity(tt:hashtag)',
    'PhoneNumber': 'Entity(tt:phone_number)',
    'EmailAddress': 'Entity(tt:email_address)',
    'URL': 'Entity(tt:url)',
    'Picture': 'Entity(tt:picture)',
    'Bool': 'Boolean'
}

UNITS = dict(C=["C", "F"],
             ms=["ms", "s", "min", "h", "day", "week", "mon", "year"],
             m=["m", "km", "mm", "cm", "mi", "in", "ft"],
             mps=["mps", "kmph", "mph"],
             kg=["kg", "g", "lb", "oz"],
             kcal=["kcal", "kJ"],
             bpm=["bpm"],
             byte=["byte", "KB", "MB", "GB", "TB"])

MAX_ARG_VALUES = 4
MAX_STRING_ARG_VALUES = 5


class PlainThingTalkGrammar(ShiftReduceGrammar):
    '''
    The grammar of ThingTalk
    '''
    
    def __init__(self, filename, grammar_include_types=True, **kw):
        super().__init__(**kw)
        self._grammar_include_types = grammar_include_types
        
        self.init_from_file(filename)
        
    def reset(self):
        queries = OrderedDict()
        actions = OrderedDict()
        functions = dict(queries=queries, actions=actions)
        self.functions = functions
        self.allfunctions = []
        self.entities = []
        self._enum_types = OrderedDict()
        self.devices = []
        self._grammar = None
    
    def _process_devices(self, devices):
        for device in devices:
            if device['kind_type'] in ('global', 'discovery', 'category'):
                continue
            self.devices.append('device:' + device['kind'])
            if device['kind'] == 'org.thingpedia.builtin.test':
                continue
            
            for function_type in ('queries', 'actions'):
                for name, function in device[function_type].items():
                    function_name = '@' + device['kind'] + '.' + name
                    paramlist = []
                    self.functions[function_type][function_name] = paramlist
                    self.allfunctions.append(function_name)
                    for argname, argtype, is_input in zip(function['args'],
                                                          function['types'] if 'types' in function else function['schema'],
                                                          function['is_input']):
                        direction = 'in' if is_input else 'out'                    
                        paramlist.append((argname, argtype, direction))
                    
                        if argtype.startswith('Array('):
                            elementtype = argtype[len('Array('):-1]
                        else:
                            elementtype = argtype
                        if elementtype.startswith('Enum('):
                            enums = elementtype[len('Enum('):-1].split(',')
                            if not elementtype in self._enum_types:
                                self._enum_types[elementtype] = enums
    
    def _process_entities(self, entities):
        for entity in entities:
            if entity['is_well_known'] == 1:
                continue
            self.entities.append((entity['type'], entity['has_ner_support']))
    
    def init_from_file(self, filename):
        self.reset()

        with open(filename, 'r') as fp:
            thingpedia = json.load(fp)
        
        self._devices = thingpedia['devices']
        self._process_devices(thingpedia['devices'])
        self._process_entities(thingpedia['entities'])

        self.complete()
    
    def complete(self):
        self.num_functions = len(self.functions['queries']) + len(self.functions['actions'])
        
        GRAMMAR = OrderedDict({
            '$input': [('$rule',),
                       ('executor', '=', '$constant_Entity(tt:username)', ':', '$rule'),
                       ('policy', '$policy'),
                       ('bookkeeping', '$bookkeeping')],
            '$bookkeeping': [('special', '$special'),
                             ('answer', '$constant_Any')],
            '$special': [(x,) for x in SPECIAL_TOKENS],
            '$rule':  [('$stream', '=>', '$action'),
                       ('$stream_join', '=>', '$action'),
                       ('now', '=>', '$table', '=>', '$action'),
                       ('now', '=>', '$action')],
            '$policy': [('true', ':', '$policy_body'),
                        ('$filter', ':', '$policy_body')],
            '$policy_body': [('now', '=>', '$policy_action'),
                             ('$policy_query', '=>', 'notify'),
                             ('$policy_query', '=>', '$policy_action')],
            '$policy_query': [('*',),
                              #('$thingpedia_device_star'),
                              ('$thingpedia_queries',),
                              ('$thingpedia_queries', 'filter', '$filter')],
            '$policy_action': [('*',),
                               #('$thingpedia_device_star'),
                               ('$thingpedia_actions',),
                               ('$thingpedia_actions', 'filter', '$filter')],
            '$table': [('$thingpedia_queries',),
                       ('(', '$table', ')', 'filter', '$filter'),
                       ('aggregate', 'min', '$out_param_Any', 'of', '(', '$table', ')'),
                       ('aggregate', 'max', '$out_param_Any', 'of', '(', '$table', ')'),
                       ('aggregate', 'sum', '$out_param_Any', 'of', '(', '$table', ')'),
                       ('aggregate', 'avg', '$out_param_Any', 'of', '(', '$table', ')'),
                       ('aggregate', 'count', 'of', '(', '$table', ')'),
                       ('aggregate', 'argmin', '$out_param_Any', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
                       ('aggregate', 'argmax', '$out_param_Any', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
                       ('$table_join',),
                       ('window', '$constant_Number', ',', '$constant_Number', 'of', '(', '$stream', ')'),
                       ('timeseries', '$constant_Date', ',', '$constant_Measure(ms)', 'of', '(', '$stream', ')'),
                       ('sequence', '$constant_Number', ',', '$constant_Number', 'of', '(', '$table', ')'),
                       ('history', '$constant_Date', ',', '$constant_Measure(ms)', 'of', '(', '$table', ')')
                       ],
            '$table_join': [('(', '$table', ')', 'join', '(', '$table', ')'),
                            ('$table_join', 'on', '$param_passing')],
            '$stream': [('timer', 'base', '=', '$constant_Date', ',', 'interval', '=', '$constant_Measure(ms)'),
                        ('attimer', 'time', '=', '$constant_Time',),
                        ('monitor', '(', '$table', ')'),
                        ('monitor', '(', '$table', ')', 'on', 'new', '$out_param_Any'),
                        ('monitor', '(', '$table', ')', 'on', 'new', '[', '$out_param_list', ']'),
                        ('edge', '(', '$stream', ')', 'on', '$filter'),
                        ('edge', '(', '$stream', ')', 'on', 'true'),
                        #('$stream_join',)
                        ],
            '$stream_join': [('(', '$stream', ')', '=>', '(', '$table', ')'),
                             ('$stream_join', 'on', '$param_passing')],
            '$action': [('notify',),
                        ('return',),
                        ('$thingpedia_actions',)],
            '$thingpedia_queries': [('$thingpedia_queries', '$const_param_list')],
            '$thingpedia_actions': [('$thingpedia_actions', '$input_param_list')],
            '$input_param': [('$const_param',),
                             ('on', '$param_passing')],
            '$input_param_list': [('$input_param',),
                                  ('$input_param_list', '$input_param')],
            '$param_passing': [('$param_name', '=', '$param_name')],
            '$const_param': [('$param_name', '=', '$constant')],
            '$out_param_list': [('$param_name',),
                                ('$out_param_list', ',', '$param_name')],

            '$filter': [('$or_filter',),
                        ('$filter', 'and', '$or_filter',)],
            '$or_filter': [('$atom_filter',),
                           ('not', '$atom_filter',),
                           ('$or_filter', 'or', '$atom_filter')
                           ],
            
            '$atom_filter': [('$thingpedia_queries', '{', 'true', '}'),
                             ('$thingpedia_queries', '{', '$filter', '}'),
                             ('$param_name', '==', '$constant'),
                             ('$param_name', '=~', '$constant'),
                             ('$param_name', '~=', '$constant'),
                             ('$param_name', 'starts_with', '$constant'),
                             ('$param_name', 'ends_with', '$constant'),
                             ('$param_name', '>=', '$constant'),
                             ('$param_name', '<=', '$constant'),
                             ('$param_name', 'contains', '$constant'),
                             ('$param_name', 'in_array', '$constant_Array')],

            '$constant_Array': [('[', '$constant_array_values', ']',)],
            '$constant_array_values': [('$constant',),
                                       ('$constant_array_values', ',', '$constant')],
            '$constant': [('undefined',)],
            '$param_name': [],

            '$word_list': [('WORD',),
                           ('$word_list', 'WORD')]
        })
        
        def add_type(type, value_rules, operators):
            assert all(isinstance(x, tuple) for x in value_rules)
            GRAMMAR['$constant'] += value_rules

        # base types
        for type, (operators, values) in TYPES.items():
            value_rules = []
            for v in values:
                if isinstance(v, tuple):
                    value_rules.append(v) 
                elif v == 'QUOTED_STRING':
                    for i in range(MAX_STRING_ARG_VALUES):
                        value_rules.append((v + '_' + str(i), ))
                elif v[0].isupper():
                    for i in range(MAX_ARG_VALUES):
                        value_rules.append((v + '_' + str(i), ))
                else:
                    value_rules.append((v,))
            add_type(type, value_rules, operators)
        for base_unit, units in UNITS.items():
            value_rules = [('$constant', 'unit:' + unit) for unit in units]
            value_rules += [('$constant', '$constant', 'unit:' + unit) for unit in units]
            operators, _ = TYPES['Number']
            add_type('Measure(' + base_unit + ')', value_rules, operators)
        for i in range(MAX_ARG_VALUES):
            GRAMMAR['$constant'].append(('DURATION_' + str(i),))

        # well known entities
        add_type('Entity(tt:device)', [(device,) for device in self.devices], ['=='])
        #add_type('Entity(tt:device)', [], ['='])

        # other entities
        for generic_entity, has_ner in self.entities:
            if has_ner:
                value_rules = [('GENERIC_ENTITY_' + generic_entity + "_" + str(i), ) for i in range(MAX_ARG_VALUES)]
                value_rules.append(('"', '$word_list', '"', '^^' + generic_entity,))
            else:
                value_rules = []
            add_type('Entity(' + generic_entity + ')', value_rules, ['=='])
            
        # maps a parameter to the list of types it can possibly have
        # over the whole Thingpedia
        param_types = OrderedDict()
        # add a parameter over the source
        param_types['source'] = OrderedSet()
        param_types['source'].add('Entity(tt:contact)')
        
        for function_type in ('queries', 'actions'):
            for function_name, params in self.functions[function_type].items():
                GRAMMAR['$thingpedia_' + function_type].append((function_name,))

        for function_type in ('queries', 'actions'):
            for function_name, params in self.functions[function_type].items():
                for param_name, param_type, param_direction in params:
                    if param_type in TYPE_RENAMES:
                        param_type = TYPE_RENAMES[param_type]
                    if param_type.startswith('Array('):
                        element_type = param_type[len('Array('):-1]
                        if element_type in TYPE_RENAMES:
                            param_type = 'Array(' + TYPE_RENAMES[element_type] + ')'
                    if param_name not in param_types:
                        param_types[param_name] = OrderedSet()
                    if self._grammar_include_types:
                        param_types[param_name].add(param_type)
                    else:
                        param_types[param_name].add('Any')
                    
        enum_values = OrderedSet()

        for param_name, options in param_types.items():
            for param_type in options:
                GRAMMAR['$param_name'].append(('param:' + param_name + ':' + param_type,))
                if param_type.startswith('Enum('):
                    enum_type = self._enum_types[param_type]
                    for enum in enum_type:
                        if enum in enum_values:
                            continue
                        
                        enum_values.add(enum)
                        GRAMMAR['$constant'].append(('enum:' + enum,))

        self._grammar = GRAMMAR
        self.construct_parser(self._grammar)

        if self._logger:
            self._logger.info('num functions', self.num_functions)
            self._logger.info('num queries', len(self.functions['queries']))
            self._logger.info('num actions', len(self.functions['actions']))
            self._logger.info('num other', len(self.tokens) - self.num_functions - self.num_control_tokens)

    def tokenize_program(self, program):
        if isinstance(program, str):
            program = program.split(' ')

        in_string = False
        for i, token in enumerate(program):
            if token == '"':
                in_string = not in_string
                yield self.dictionary[token], None
                continue
            elif in_string:
                yield self._word_id, token
                continue

            if (not self._grammar_include_types) and token.startswith('param:'):
                token = 'param:' + token.split(':')[1] + ':Any'
            
            if token not in self.dictionary:
                raise ValueError("Invalid token " + token)
            else:
                yield self.dictionary[token], None

        
if __name__ == '__main__':
    grammar = PlainThingTalkGrammar(sys.argv[1], grammar_include_types=False)
    for t in grammar.tokens:
        print(t)
