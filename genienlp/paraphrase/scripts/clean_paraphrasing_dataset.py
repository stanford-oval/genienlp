from argparse import ArgumentParser
import csv
import sys
from tqdm import tqdm
from genienlp.util import detokenize
import random
import os

csv.field_size_limit(sys.maxsize)

def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def remove_quotation(s):
    s = s.replace('``', '')
    s = s.replace('\'\'', '')
    s = s.replace('"', '')
    if s.startswith('\''):
        s = s[1:]
    if s.endswith('\''):
        s = s[:-1]
    return s

def is_valid(s):
    return 'http' not in s and s.count('-') <= 4 and s.count('.') <= 4 and is_english(s) \
        and '_' not in s and '%' not in s and '/' not in s and '*' not in s and '\\' not in s \
            and 'www' not in s and sum(c.isdigit() for c in s) <= 10 and s.count('(') == s.count(')')

def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=str,
                        help='The path to the input .tsv file.')
    parser.add_argument('output_dir', type=str,
                        help='The path to the folder to save train.tsv and dev.tsv files.')

    parser.add_argument('--train_ratio', type=float, required=True,
                        help='The ratio of input examples that go to the training set')
    parser.add_argument('--seed', default=123, type=int, help='Random seed used for train/dev split.')

    # By default, we swap the columns so that the target of paraphrasing will be a grammatically correct sentence, i.e. written by a human, not an NMT
    parser.add_argument('--first_columns', type=int, nargs='+', default=[1], help='The column indices in the input file to put in the first column of the output file')
    parser.add_argument('--second_columns', type=int, nargs='+', default=[0], help='The column indices in the input file to put in the second column of the output file')
    
    parser.add_argument('--min_length', type=int, default=30, help='Minimum number of characters that each phrase should have in order to be included')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum number of characters that each phrase should have in order to be included')
    parser.add_argument('--skip_check', action='store_true', help='Skip validity check.')
    parser.add_argument('--skip_normalization', action='store_true', help='Do not remove quotation marks or detokenize.')
    parser.add_argument('--lower_case', action='store_true', help='Convert everything to lower case.')
    parser.add_argument('--max_examples', type=int, default=1e10, help='Maximum number of examples in the output.')


    args = parser.parse_args()
    random.seed(args.seed)

    drop_count = 0
    # number_of_lines = get_number_of_lines(args.input)
    # number_of_lines = get_number_of_lines(args.input)
    output_size = 0
    with open(args.input, 'r') as input_file, \
        open(os.path.join(args.output_dir, 'train.tsv'), 'w') as train_output_file, \
        open(os.path.join(args.output_dir, 'dev.tsv'), 'w') as dev_output_file:
        train_writer = csv.writer(train_output_file, delimiter='\t')
        dev_writer = csv.writer(dev_output_file, delimiter='\t')
        reader = csv.reader(input_file, delimiter='\t')
        for row in tqdm(reader, desc='Lines'):
            is_written = False

            # Decide which output file this example should be written to. Note that all the examples generated from this row will go to the same file
            r = random.random()
            if r < args.train_ratio:
                writer = train_writer
            else:
                writer = dev_writer

            for first_column in args.first_columns:
                if first_column >= len(row):
                    continue
                for second_column in args.second_columns:
                    if second_column >= len(row) or first_column == second_column:
                        continue
                    first = row[first_column] # input sequence
                    second = row[second_column] # output_sequence
                    # print('first = ', first)
                    # print('second = ', second)
                    if not args.skip_check and \
                        (len(first) < args.min_length or len(second) < args.min_length \
                            or len(first) > args.max_length or len(second) > args.max_length \
                            or not is_valid(first) or not is_valid(second)):
                        drop_count += 1
                        continue
                    if not args.skip_normalization:
                        first = remove_quotation(detokenize(first))
                        second = remove_quotation(detokenize(second))
                    first = first.strip()
                    second = second.strip()
                    if args.lower_case:
                        first = first.lower()
                        second = second.lower()
                    if first.lower() == second.lower() or first == '' or second == '':
                        drop_count += 1
                        continue
                    writer.writerow([first, second])
                    is_written = True
            if is_written:
                output_size += 1
                if output_size >= args.max_examples:
                    break

    print('Dropped', drop_count, 'examples')

if __name__ == '__main__':
    main()
