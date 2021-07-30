import os
import sys

import ujson

data_folder = sys.argv[1]
file_name = sys.argv[2]
final_data = {}

i = 0

for file in os.listdir(data_folder):
    if file.startswith(file_name):
        full_path = os.path.join(data_folder, file)
        with open(full_path, 'r') as fin:
            data = ujson.load(fin)
        print(f'processed file {i}')
        i += 1
        final_data.update(data)

with open(os.path.join(data_folder, f'{file_name}_final.json'), 'w') as fout:
    ujson.dump(final_data, fout)
