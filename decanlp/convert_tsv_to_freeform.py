from argparse import ArgumentParser
import csv
from tqdm import tqdm
# from .util import get_number_of_lines

def detokenize(text):
    tokens = ["'d", "n't", "'ve", "'m", "'re", "'ll", ".", ",", "?", "'s", ")"]
    for t in tokens:
        text = text.replace(' ' + t, t)
    text = text.replace("( ", "(")
    return text

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

    args = parser.parse_args()

    drop_count = 0
    # number_of_lines = get_number_of_lines(args.input)
    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        reader = csv.reader(input_file, delimiter='\t')
        for row in tqdm(reader, desc='Lines'):
            if 'http' in row[0]:
                drop_count += 1
                continue
            output_file.write(detokenize(row[1]) + args.start_special_token +
                              detokenize(row[0]) + args.end_special_token + '\n')  # we swap the columns so that the target of paraphrasing will be a grammatically correct sentence
    print('Dropped', drop_count, 'examples')

if __name__ == '__main__':
    main()
