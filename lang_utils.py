

def get_functions(program):
    return [x for x in program.split(' ') if x.startswith('@')]