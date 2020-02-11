from argparse import ArgumentParser
import csv
from tqdm import tqdm
import random

def remove_thingtalk_quotes(thingtalk):
    while True:
        # print('before: ', thingtalk)
        l1 = thingtalk.find('"')
        if l1 < 0:
            break
        l2 = thingtalk.find('"', l1+1)
        assert l2 >= 0
        thingtalk = thingtalk[:l1] + '<temp>' + thingtalk[l2+1:]
        # print('after: ', thingtalk)
    thingtalk = thingtalk.replace('<temp>', '""')
    return thingtalk

def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=str,
                        help='The path to the input.')
    parser.add_argument('output', type=str,
                        help='The path to the output file.')
    parser.add_argument('--query_file', type=str,
                        help='The path to the file containing new queries.')
    parser.add_argument('--transformation', type=str, choices=['remove_thingtalk_quotes', 'replace_queries', 'none'], default='none',
                        help='The type of transformation to apply.')
    parser.add_argument('--output_columns', type=int, nargs='+', default=[0, 1, 2],
                        help='The type of transformation to apply.')

    args = parser.parse_args()

    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        reader = csv.reader(input_file, delimiter='\t')
        if args.transformation == 'replace_queries':
            new_queries = []
            query_file = open(args.query_file, 'r')
            for q in query_file:
                new_queries.append(q)
        line_count = 0
        for row in tqdm(reader, desc='Lines'):
            query = row[1]
            thingtalk = row[2]
            if args.transformation == 'remove_thingtalk_quotes':
                row[2] = remove_thingtalk_quotes(thingtalk)
            if args.transformation == 'replace_queries':
                row[1] = new_queries[line_count]
            output_row = ""
            for i, column in enumerate(args.output_columns):
                output_row += row[column]
                if i < len(args.output_columns)-1:
                    output_row += '\t'
            output_file.write(output_row + '\n')
            line_count += 1

if __name__ == '__main__':
    main()
