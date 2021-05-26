import csv
import os
import random
import re
import sys

from ...data_utils.progbar import progress_bar
from ...util import detokenize

csv.field_size_limit(sys.maxsize)


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def normalize(s):
    # remove quotations
    s = s.replace('``', '')
    s = s.replace('\'\'', '')
    s = s.replace('"', '')
    if s.startswith('\''):
        s = s[1:]
    if s.endswith('\''):
        s = s[:-1]

    s = s.replace('...', '')
    s = s.replace(',', ', ')
    s = re.sub('\s\s+', ' ', s)  # remove duplicate white spaces
    s = s.strip()

    return s


def is_valid(s):
    return (
        'http' not in s
        and s.count('-') <= 4
        and s.count('.') <= 4
        and is_english(s)
        and '_' not in s
        and '%' not in s
        and '/' not in s
        and '[' not in s
        and ']' not in s
        and '*' not in s
        and '\\' not in s
        and 'www' not in s
        and sum(c.isdigit() for c in s) <= 10
        and s.count('(') == s.count(')')
    )


def normalized_levenshtein(s1, s2, mode='character'):
    if mode != 'character' and isinstance(s1, str):
        s1 = s1.split()
        s2 = s2.split()
    if len(s1) < len(s2):
        return normalized_levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1] / max(len(s1), len(s2))


def pos_tag_string(sentence: str):
    # load NLTK lazily
    import nltk

    nltk.download('averaged_perceptron_tagger', quiet=True)
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(sentence))
    tags = [t[1] for t in tagged_tokens]
    return ' '.join(tags).lower()


def parse_argv(parser):
    parser.add_argument('input', type=str, help='The path to the input .tsv file.')
    parser.add_argument('output_dir', type=str, help='The path to the folder to save train.tsv and dev.tsv files.')

    parser.add_argument(
        '--train_ratio', type=float, default=0.95, help='The ratio of input examples that go to the training set'
    )
    parser.add_argument('--seed', default=123, type=int, help='Random seed used for train/dev split.')

    # By default, we swap the columns so that the target of paraphrasing will be a grammatically correct sentence, i.e. written by a human, not an NMT
    parser.add_argument(
        '--first_columns',
        type=int,
        nargs='+',
        default=[2],
        help='The column indices in the input file to put in the first column of the output file',
    )
    parser.add_argument(
        '--second_columns',
        type=int,
        nargs='+',
        default=[1],
        help='The column indices in the input file to put in the second column of the output file',
    )

    parser.add_argument(
        '--min_length',
        type=int,
        default=20,
        help='Minimum number of characters that each phrase should have in order to be included',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=200,
        help='Maximum number of characters that each phrase should have in order to be included',
    )
    parser.add_argument('--edit_distance_mode', type=str, default='none', choices=['character', 'word', 'none'])
    parser.add_argument(
        '--min_edit_distance',
        type=float,
        default=0.0001,
        help='We will not include phrase pairs that have a normalized edit distance below this number.',
    )
    parser.add_argument(
        '--plot_edit_distance', action='store_true', help='Save a plot of the normalized edit distance distribution.'
    )
    parser.add_argument('--skip_check', action='store_true', help='Skip validity check.')
    parser.add_argument('--skip_normalization', action='store_true', help='Do not remove quotation marks or detokenize.')
    parser.add_argument('--lower_case', action='store_true', help='Convert everything to lower case.')
    parser.add_argument(
        '--prepend_pos_tags',
        action='store_true',
        help='Prepend the part-of-speech tag of the output to the beginning of the input.',
    )
    parser.add_argument(
        '--sep_token', type=str, default='</s>', help='Token to insert between the part-of-speech tag and the input sentence.'
    )
    parser.add_argument('--max_examples', type=int, default=1e10, help='Maximum number of examples in the output.')


def main(args):
    random.seed(args.seed)

    drop_count = 0
    # number_of_lines = get_number_of_lines(args.input)
    # number_of_lines = get_number_of_lines(args.input)
    output_size = 0
    included_examples = 0
    sum_edit_distance = 0
    all_normalized_edit_distances = []
    with open(args.input, 'r') as input_file, open(os.path.join(args.output_dir, 'train.tsv'), 'w') as train_output_file, open(
        os.path.join(args.output_dir, 'dev.tsv'), 'w'
    ) as dev_output_file:
        train_writer = csv.writer(train_output_file, delimiter='\t')
        dev_writer = csv.writer(dev_output_file, delimiter='\t')
        reader = csv.reader(input_file, delimiter='\t')
        for row in progress_bar(reader, desc='Lines'):
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
                    first = row[first_column]  # input sequence
                    second = row[second_column]  # output_sequence
                    # print('first = ', first)
                    # print('second = ', second)
                    if not args.skip_check and (
                        len(first) < args.min_length
                        or len(second) < args.min_length
                        or len(first) > args.max_length
                        or len(second) > args.max_length
                        or not is_valid(first)
                        or not is_valid(second)
                    ):
                        drop_count += 1
                        continue
                    if not args.skip_normalization:
                        first = normalize(detokenize(first))
                        second = normalize(detokenize(second))
                    first = first.strip()
                    second = second.strip()
                    if args.lower_case:
                        first = first.lower()
                        second = second.lower()
                    if first == '' or second == '' or first.lower() == second.lower():
                        drop_count += 1
                        continue
                    if args.edit_distance_mode != 'none':
                        normalized_edit_distance = normalized_levenshtein(
                            first.lower(), second.lower(), mode=args.edit_distance_mode
                        )
                        # print('normalized_edit_distance = ', normalized_edit_distance)
                        if normalized_edit_distance < args.min_edit_distance:
                            drop_count += 1
                            continue
                        all_normalized_edit_distances.append(normalized_edit_distance)
                        sum_edit_distance += normalized_edit_distance
                    if args.prepend_pos_tags:
                        first = pos_tag_string(second) + ' ' + args.sep_token + ' ' + first
                    writer.writerow([first, second])
                    output_size += 1
                    is_written = True
            if is_written:
                included_examples += 1
                if included_examples >= args.max_examples:
                    break

    print('Dropped', drop_count, 'examples')
    print('Average normalized edit distance between pairs is ', sum_edit_distance / output_size)
    if args.edit_distance_mode != 'none' and args.plot_edit_distance:
        import matplotlib.pyplot as plt

        _, _, _ = plt.hist(all_normalized_edit_distances, 20)
        plt.savefig(os.path.join(args.output_dir, 'edit_distance_plot.png'))
