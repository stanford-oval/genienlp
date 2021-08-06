import argparse
from collections import Counter

import jsonlines
from termcolor import colored

from genienlp.ned.bootleg import BatchBootlegEntityDisambiguator

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--database_dir', type=str)
parser.add_argument('--ned_domains', type=str, nargs='+')
parser.add_argument('--subsample', type=int, default='1000000000')


args = parser.parse_args()

if __name__ == '__main__':

    args.bootleg_model = 'bootleg_uncased_mini'
    args.ned_normalize_types = 'yes'
    args.bootleg_output_dir = 'results_temp'
    args.embeddings = '.embeddings'
    args.almond_type_mapping_path = None
    args.bootleg_prob_threshold = 0.3
    args.max_types_per_qid = 1
    args.max_qids_per_entity = 1
    args.max_features_size = args.max_types_per_qid * args.max_qids_per_entity

    bootleg = BatchBootlegEntityDisambiguator(args)

    all_typeqids = []
    all_aliases = []
    all_qids = []
    all_probs = []
    unknown_qids = set()

    lines = jsonlines.open(args.input_file, 'r')

    all_titles = Counter()
    all_new_titles = Counter()

    fout = open('results.txt', 'w')

    for count, bootleg_line in enumerate(lines):
        if count >= args.subsample:
            break

        for alias, all_qids, all_probs, span in zip(
            bootleg_line['aliases'], bootleg_line['cands'], bootleg_line['cand_probs'], bootleg_line['spans']
        ):
            # without mapping
            bootleg.args.ned_normalize_types = 'no'
            type_ids, type_probs, qids = bootleg.collect_features_per_alias(alias, all_probs, all_qids)
            type_vocabs = []
            for id_ in type_ids:
                type_vocab = bootleg.typeqid_to_type_vocab[bootleg.id2typeqid[id_]]
                type_vocabs.append(type_vocab)
                all_titles[type_vocab] += 1
            bootleg.args.ned_normalize_types = 'yes'
            type_ids, type_probs, qids = bootleg.collect_features_per_alias(alias, all_probs, all_qids)
            for id_ in type_ids:
                all_new_titles[bootleg.typeqid_to_type_vocab[bootleg.id2typeqid[id_]]] += 1
            type_vocabs_new = []
            for id_ in type_ids:
                type_vocab = bootleg.typeqid_to_type_vocab[bootleg.id2typeqid[id_]]
                type_vocabs_new.append(type_vocab)
                all_titles[type_vocab] += 1

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
