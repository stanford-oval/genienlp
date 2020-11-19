import csv
import re

from ...util import tokenize, lower_case, remove_thingtalk_quotes
from ...data_utils.progbar import progress_bar

def is_subset(set1, set2):
    """
    Returns True if set1 is a subset of or equal to set2
    """
    return all([e in set2 for e in set1])

def passes_heuristic_checks(row, args):
    all_input_columns = ' '.join([row[c] for c in args.input_columns])
    input_special_tokens = set(re.findall('[A-Za-z:_.]+_[0-9]', all_input_columns))
    output_special_tokens = set(re.findall('[A-Za-z:_.]+_[0-9]', row[args.thingtalk_column]))
    if not is_subset(output_special_tokens, input_special_tokens):
        return False
    _, quote_values = remove_thingtalk_quotes(row[args.thingtalk_column])
    if quote_values is None:
        return False # ThingTalk is not syntactic, so remove this example
    for q in quote_values:
        if q not in all_input_columns:
            return False
    return True
    
def parse_argv(parser):
    parser.add_argument('input', type=str,
                        help='The path to the input file.')
    parser.add_argument('output', type=str,
                        help='The path to the output file.')
    parser.add_argument('--query_file', type=str,
                        help='The path to the file containing new queries.')
    parser.add_argument('--thrown_away', type=str, default=None,
                        help='The path to the output file that will contain inputs that were removed because of `--transformation`.')
    parser.add_argument('--thingtalk_gold_file', type=str,
                        help='The path to the file containing the dataset with a correct thingtalk column.')
    parser.add_argument('--num_new_queries', type=int, default=1,
                        help='Number of new queries per old query. Valid if "--transformation replace_queries" is used.')
    parser.add_argument('--transformation', type=str, choices=['remove_thingtalk_quotes',
                                                                'replace_queries',
                                                                'remove_wrong_thingtalk',
                                                                'get_wrong_thingtalk',
                                                                'merge_input_file_with_query_file',
                                                                'none'], default='none', help='The type of transformation to apply.')
    parser.add_argument('--remove_duplicates', action='store_true',
                        help='Remove duplicate natural utterances. Note that this also removes cases where a natural utterance has multiple ThingTalk codes.')

    # These arguments are effective only if --transformation=get_wrong_thingtalk
    parser.add_argument('--remove_with_heuristics', action='store_true',
                        help='Remove examples if the values inside quotations in ThingTalk have changed or special words like NUMBER_0 cannot be found in TT anymore.')
    parser.add_argument('--replace_with_gold', action='store_true', help='Instead of the original ThingTalk, output what the parser said is gold.')

    parser.add_argument('--task', type=str, required=True, choices=['almond', 'almond_dialogue_nlu', 'almond_dialogue_nlu_agent'],
                        help='Specifies the meaning of columns in the input file and the ones that should go to the output')

    # parser.add_argument('--output_columns', type=int, nargs='+', default=None,
    #                     help='The columns to write to output. By default, we output all columns.')
    # parser.add_argument('--id_column', type=int, default=0,
    #                     help='The column index in the input file that contains the unique id')
    # parser.add_argument('--utterance_column', type=int, default=1,
    #                     help='The column index in the input file that contains the natural utterance')
    # parser.add_argument('--thingtalk_column', type=int, default=2,
    #                     help='The column index in the input file that contains the ThingTalk code.')
    # parser.add_argument('--no_duplication_columns', type=int, nargs='+', default=None,
    #                     help='The columns indices in the input file that determine whether two rows are duplicates of each other or not.')


def main(args):
    if args.task == 'almond':
        args.id_column = 0
        args.utterance_column = 1
        args.thingtalk_column = 2
        args.output_columns = [0, 1, 2]
        args.no_duplication_columns = [1]
        args.input_columns = [1]

    elif args.task == 'almond_dialogue_nlu' or 'almond_dialogue_nlu_agent':
        args.id_column = 0
        # column 1 is ontext (ThingTalk)
        args.utterance_column = 2
        args.thingtalk_column = 3
        args.output_columns = [0, 1, 2, 3]
        args.no_duplication_columns = [1, 2]
        args.input_columns = [1, 2]

    if args.output_columns is None:
        # if args.transformation == 'remove_wrong_thingtalk':
            # args.output_columns = [0, 1, 2, 3, 4] # we add original utterance and ThingTalk as well
        args.output_columns = [0, 1, 2]
    if args.remove_duplicates and args.no_duplication_columns is None:
        raise ValueError('You should specify columns that define duplication')

    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        reader = csv.reader(input_file, delimiter='\t')
        if args.transformation in ['replace_queries', 'merge_input_file_with_query_file']:
            new_queries = []
            query_file = open(args.query_file, 'r')
            for q in query_file:
                new_queries.append(lower_case(tokenize(q.strip())))
            new_query_count = 0
        if args.transformation in ['remove_wrong_thingtalk', 'get_wrong_thingtalk']:
            gold_thingtalks = []
            thingtalk_gold_file_reader = csv.reader(open(args.thingtalk_gold_file, 'r'), delimiter='\t')
            for q in thingtalk_gold_file_reader:
                gold_thingtalks.append(q[1].strip())
            gold_thingtalk_count = 0
            
        duplicate_count = 0
        heuristic_count = 0
        written_count = 0
        if args.remove_duplicates:
            seen_examples = set()
        all_thrown_away_rows = []
        for row in progress_bar(reader, desc='Lines'):
            output_rows = []
            thrown_away_rows = []
            if args.transformation == 'remove_thingtalk_quotes':
                row[args.thingtalk_column], _ = remove_thingtalk_quotes(row[args.thingtalk_column])
            if args.transformation == 'remove_wrong_thingtalk':
                if row[args.thingtalk_column] == gold_thingtalks[gold_thingtalk_count]:
                    output_rows = [row]
                else:
                    output_rows = []
                    thrown_away_rows = [row]
                gold_thingtalk_count += 1
            elif args.transformation == 'get_wrong_thingtalk':
                if row[args.thingtalk_column] != gold_thingtalks[gold_thingtalk_count]:
                    if args.replace_with_gold:
                        row[args.thingtalk_column] = gold_thingtalks[gold_thingtalk_count]
                    output_rows = [row]
                else:
                    output_rows = []
                gold_thingtalk_count += 1
            elif args.transformation == 'merge_input_file_with_query_file':
                output_rows.append(row)
                for _ in range(args.num_new_queries):
                    row = row.copy()
                    row[args.utterance_column] = new_queries[new_query_count]
                    output_rows.append(row)
                    new_query_count += 1
            elif args.transformation == 'replace_queries':
                for idx in range(args.num_new_queries):
                    copy_row = row.copy()
                    copy_row[args.utterance_column] = new_queries[new_query_count]
                    copy_row[args.id_column] = 'A' + copy_row[args.id_column] + '-' + str(idx) # add 'A' for auto-paraphrasing
                    output_rows.append(copy_row)
                    new_query_count += 1
            else:
                assert args.transformation == 'none'
                # do basic clean-up because old generation code missed these
                row[args.utterance_column] = row[args.utterance_column].replace('<pad>', '')
                row[args.utterance_column] = re.sub('\s\s+', ' ', row[args.utterance_column])
                row[args.utterance_column] = row[args.utterance_column].strip()
                output_rows = [row]
            
            for o in output_rows:
                output_row = ""
                if args.remove_with_heuristics:
                    if not passes_heuristic_checks(o, args):
                        heuristic_count += 1
                        continue
                if args.remove_duplicates:
                    normalized_example = re.sub('\s+', '', ''.join([o[i] for i in args.no_duplication_columns]))
                    # print(normalized_example)
                    if normalized_example in seen_examples:
                        duplicate_count += 1
                        continue
                    else:
                        seen_examples.add(normalized_example)
                written_count += 1
                for i, column in enumerate(args.output_columns):
                    output_row += o[column]
                    if i < len(args.output_columns)-1:
                        output_row += '\t'
                output_file.write(output_row + '\n')
            for o in thrown_away_rows:
                if not args.remove_with_heuristics or (args.remove_with_heuristics and passes_heuristic_checks(o, args)):
                    all_thrown_away_rows.append(o)

        if args.thrown_away is not None:
            # write the thrown away exampels into a file
            with open(args.thrown_away, 'w') as output_file:
                for o in all_thrown_away_rows:
                    output_row = ""
                    for i, column in enumerate(args.output_columns):
                        output_row += o[column]
                        if i < len(args.output_columns)-1:
                            output_row += '\t'
                    output_file.write(output_row + '\n')

        print('Removed %d duplicate rows' % duplicate_count)
        print('Removed %d rows because the thingtalk quote did not satisfy heuristic rules' % heuristic_count)
        print('Output size is %d rows' % written_count)
