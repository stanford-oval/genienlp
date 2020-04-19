from argparse import ArgumentParser
import csv
import sys
from tqdm import tqdm
from genienlp.util import detokenize, get_number_of_lines

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
    parser.add_argument('output', type=str,
                        help='The path to the output .txt file.')

    # By default, we swap the columns so that the target of paraphrasing will be a grammatically correct sentence, i.e. written by a human, not an NMT
    parser.add_argument('--first_column', type=int, default=1, help='The column index in the input file to put in the first column of the output file')
    parser.add_argument('--second_column', type=int, default=0, help='The column index in the input file to put in the second column of the output file')
    
    parser.add_argument('--min_length', type=int, default=30, help='Minimum number of characters that each phrase should have in order to be included')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum number of characters that each phrase should have in order to be included')
    parser.add_argument('--skip_check', action='store_true', help='Skip validity check.')
    parser.add_argument('--skip_normalization', action='store_true', help='Do not remove quotation marks or detokenize.')
    parser.add_argument('--lower_case', action='store_true', help='Convert everything to lower case.')
    parser.add_argument('--max_output_size', type=int, default=1e10, help='Maximum number of examples in the output.')

    args = parser.parse_args()

    drop_count = 0
    # number_of_lines = get_number_of_lines(args.input)
    # number_of_lines = get_number_of_lines(args.input)
    output_size = 0
    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        reader = csv.reader(input_file, delimiter='\t')
        for row in tqdm(reader, desc='Lines'):
            first = row[args.first_column] # input sequence
            second = row[args.second_column] # output_sequence
            # print(first)
            # print(second)
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
            output_size += 1
            if output_size >= args.max_output_size:
                break
    print('Dropped', drop_count, 'examples')

if __name__ == '__main__':
    main()
