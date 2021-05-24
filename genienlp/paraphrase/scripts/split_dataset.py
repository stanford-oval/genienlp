import random

from ...data_utils.progbar import progress_bar


def parse_argv(parser):
    parser.add_argument('input', type=str, help='The path to the input.')
    parser.add_argument('output1', type=str, help='The path to the output train file.')
    parser.add_argument('output2', type=str, help='The path to the output dev file.')
    parser.add_argument('--output1_ratio', type=float, required=True, help='The ratio of input examples that go to output1')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')


def main(args):
    random.seed(args.seed)

    with open(args.input, 'r') as input_file, open(args.output1, 'w') as output_file1, open(args.output2, 'w') as output_file2:
        for line in progress_bar(input_file):
            r = random.random()
            if r < args.output1_ratio:
                output_file1.write(line)
            else:
                output_file2.write(line)
