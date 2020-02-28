from argparse import ArgumentParser
import csv
from tqdm import tqdm
from .util import detokenize

def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def is_valid(s):
    return 'http' not in s and s.count('-') <= 4 and s.count('.') <= 4 and is_english(s) \
        and '_' not in s and '/' not in s and '*' not in s and '\\' not in s \
            and 'www' not in s and sum(c.isdigit() for c in s) <= 10

def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=str,
                        help='The path to the input .tsv file.')
    parser.add_argument('output', type=str,
                        help='The path to the output .txt file.')
    parser.add_argument('--start_special_token', type=str, default='<paraphrase>',
                        help='The special token for the start of paraphrases.')
    parser.add_argument('--end_special_token', type=str, default='</paraphrase>',
                        help='The special token for the end of paraphrases.')
    parser.add_argument('--first_column', type=int, default=1, help='The column index in the input file to put in the first column of the output file')
    parser.add_argument('--second_column', type=int, default=0, help='The column index in the input file to put in the second column of the output file')
    parser.add_argument('--skip_check', action='store_true', help='Skip validity check.')

    args = parser.parse_args()

    drop_count = 0
    # number_of_lines = get_number_of_lines(args.input)
    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        reader = csv.reader(input_file, delimiter='\t')
        for row in tqdm(reader, desc='Lines'):
            if not args.skip_check and (not is_valid(row[args.first_column]) or not is_valid(row[args.second_column])):
                drop_count += 1
                continue
            output_file.write(detokenize(row[args.first_column]) + args.start_special_token +
                              detokenize(row[args.second_column]) + args.end_special_token + '\n')  # we swap the columns so that the target of paraphrasing will be a grammatically correct sentence
    print('Dropped', drop_count, 'examples')

if __name__ == '__main__':
    main()
