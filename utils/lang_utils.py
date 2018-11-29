import os
import numpy

def get_functions(program):
    return [x for x in program.split(' ') if x.startswith('@')]

def get_devices(program):
    return [x.rsplit('.', 1)[0] for x in program.split(' ') if x.startswith('@')]