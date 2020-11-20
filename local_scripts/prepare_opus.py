import sys
from collections import defaultdict

id_file, output_id_file = sys.argv[1:3]

DEFAULT_ID = 'tatoeba'

id2size = defaultdict(int)
with open(id_file, 'r') as fin, open(output_id_file, 'w') as fout:
    for line in fin:
        parts = list(map(lambda part: part.strip(), line.split('\t')))
        
        # dev and test set
        if len(parts) == 2:
            main_id = DEFAULT_ID
        else:
            main_id = parts[0]

        fout.write(main_id + '/' + str(id2size[main_id]) + '\n')
        id2size[main_id] += 1
        