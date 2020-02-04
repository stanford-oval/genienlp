from argparse import ArgumentParser
import csv
from tqdm import tqdm


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

    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        reader = csv.reader(input_file, delimiter='\t')
        for row in tqdm(reader):
            output_file.write(row[0] + args.start_special_token +
                              row[1] + args.end_special_token + '\n')


if __name__ == '__main__':
    main()
