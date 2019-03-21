#
# Copyright (c) 2019, The Board of Trustees of the Leland Stanford Junior University
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

import os
import numpy
import logging
import json


logger = logging.getLogger(__name__)

def get_functions(program):
    return [x for x in program.split(' ') if x.startswith('@')]

def get_devices(program):
    return [x.rsplit('.', 1)[0] for x in program.split(' ') if x.startswith('@')]

def generate_tokens(type):

    args = type[len('Enum')+1:-1].split(',')
    result = list(map(lambda arg: f'enum:{arg}', args))
    return result

def extract_words(thingpedia):

    words_list = set()
    with open(thingpedia, 'r') as f:
        result = json.load(f)
        output = dict()

        output['devices'] = []
        output['entities'] = []

        all_devices = result['devices']
        all_entities = result['entities']

        for entity in all_entities:
            if entity['has_ner_support']:
                words_list.add('^^' + entity['type'])

        for device in all_devices:
            if device['kind_type'] in ('global', 'category', 'discovery'):
                continue

            if not device.get('kind_canonical', None):
                logger.warning('WARNING: missing canonical for device:%s' % (device['kind']))

            for function_type in ('triggers', 'queries', 'actions'):
                for function_name, function in device[function_type].items():
                    if not function['canonical']:
                        logger.warning('WARNING: missing canonical for @%s.%s' % (device['kind'], function_name))
                    else:
                        words_list.add('@' + device['kind'] + '.' + function_name)
                        for arg, type in zip(function['args'], function['types']):
                            if type.startswith('Enum'):
                                tokens = generate_tokens(type)
                                words_list.update(tokens)
                            words_list.add('param:' + arg + ':' + type)

    return words_list