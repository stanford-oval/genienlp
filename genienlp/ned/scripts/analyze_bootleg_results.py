from collections import Counter

import jsonlines
from termcolor import colored

from genienlp.ned.bootleg import BatchBootlegEntityDisambiguator


def parse_argv(parser):

    parser.add_argument('--input_file', type=str)
    parser.add_argument('--database_dir', type=str)
    parser.add_argument('--ned_domains', type=str, nargs='+')
    parser.add_argument('--subsample', type=int, default='1000000000')
    parser.add_argument('--output_file', type=str, default='results.txt')
    parser.add_argument('--bootleg_model', type=str, default='bootleg_uncased_mini')
    parser.add_argument('--bootleg_output_dir', type=str, default='results_temp')
    parser.add_argument('--embeddings', type=str, default='.embeddings')
    parser.add_argument('--almond_type_mapping_path', type=str)


def main(args):

    # ned_normalize_types = 'soft'
    # args.ned_normalize_types = 'soft'
    # args.bootleg_prob_threshold = 0.3
    # args.max_types_per_qid = 1
    # args.max_qids_per_entity = 1

    ned_normalize_types = 'soft'
    args.ned_normalize_types = ned_normalize_types
    args.bootleg_prob_threshold = 0.01
    args.max_types_per_qid = 2
    args.max_qids_per_entity = 2

    args.max_features_size = args.max_types_per_qid * args.max_qids_per_entity

    bootleg = BatchBootlegEntityDisambiguator(args)

    lines = jsonlines.open(args.input_file, 'r')

    all_titles = Counter()
    all_new_titles = Counter()

    fout = open(args.output_file, 'w')

    for count, bootleg_line in enumerate(lines):
        if count >= args.subsample:
            break

        for alias, all_qids, all_probs, span in zip(
            bootleg_line['aliases'], bootleg_line['cands'], bootleg_line['cand_probs'], bootleg_line['spans']
        ):
            # without mapping
            bootleg.args.ned_normalize_types = 'off'
            type_ids, type_probs, qids = bootleg.collect_features_per_alias(alias, all_probs, all_qids)
            type_vocabs = []
            for id_ in type_ids:
                type_vocab = bootleg.typeqid_to_type_vocab[bootleg.id2typeqid[id_]]
                type_vocabs.append(type_vocab)
                all_titles[type_vocab] += 1

            # with mapping
            bootleg.args.ned_normalize_types = ned_normalize_types
            type_ids, type_probs, qids = bootleg.collect_features_per_alias(alias, all_probs, all_qids)
            type_vocabs_new = []
            for id_ in type_ids:
                type_vocab = bootleg.typeqid_to_type_vocab[bootleg.id2typeqid[id_]]
                type_vocabs_new.append(type_vocab)
                all_new_titles[type_vocab] += 1

            if len(type_probs) == len(type_vocabs) == 0:
                continue
            fout.write(f'{alias}, {type_probs}, {qids}, {type_vocabs}, {type_vocabs_new}' + '\n')

    fout.close()

    print(f'all_titles: {all_titles.most_common()}')
    almond_type_vocabs = []
    extra_type_vocabs = []

    for tup in all_new_titles.most_common():
        if tup[0] in bootleg.almond_type_mapping:
            almond_type_vocabs.append(tup)
        else:
            extra_type_vocabs.append(tup)

    print('all_new_titles:', *[colored(tup, "red") for tup in almond_type_vocabs], *extra_type_vocabs)
