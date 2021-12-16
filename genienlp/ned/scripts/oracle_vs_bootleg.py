from collections import Counter, defaultdict

import jsonlines
from tqdm import tqdm

from genienlp.ned import BatchBootlegEntityDisambiguator


def parse_argv(parser):

    parser.add_argument('--bootleg_labels', type=str)
    parser.add_argument('--oracle_labels', type=str)
    parser.add_argument('--database_dir', type=str)
    parser.add_argument('--ned_domains', type=str, nargs='+')
    parser.add_argument('--output_file', type=str, default='comparison.txt')
    parser.add_argument('--bootleg_model', type=str, default='bootleg_uncased_mini')
    parser.add_argument('--bootleg_output_dir', type=str, default='results_temp')
    parser.add_argument('--embeddings', type=str, default='.embeddings')
    parser.add_argument('--almond_type_mapping_path', type=str)
    parser.add_argument('--ned_normalize_types', type=str, choices=['no', 'soft', 'strict'], default='strict')

def main(args):

    args.root = '.'
    #args.bootleg_prob_threshold = 0.01
    args.bootleg_prob_threshold = 0.3
    args.max_types_per_qid = 2
    args.max_qids_per_entity = 1

    args.max_features_size = args.max_types_per_qid * args.max_qids_per_entity

    bootleg = BatchBootlegEntityDisambiguator(args)

    n_lines = 0
    with open(args.bootleg_labels) as fp:
        for line in fp:
            n_lines += 1

    bootleg_lines = jsonlines.open(args.bootleg_labels, 'r')
    oracle_lines = jsonlines.open(args.oracle_labels, 'r')

    # the entity was there in the oracle and bootleg detected it with the right type
    strict_true_positive = Counter()
    # the entity was there in the oracle and bootleg detected it with the wrong type
    # (this is a false negative when ned_normalize_types == strict
    # and a true positive when ned_normalize_types === soft)
    soft_true_positive = Counter()
    # bootleg did not the detect an entity that was in the oracle
    false_negative = Counter()

    # same as soft_true_positive, but indexed by bootleg_type not oracle_type
    precision_true_positive = Counter()
    # bootleg detected the entity but the entity was not there in the oracle
    false_positive = Counter()

    mismatched_types = defaultdict(Counter)
    mispredicted_entities = defaultdict(Counter)

    for oracle_line, bootleg_line in tqdm(zip(oracle_lines, bootleg_lines), total=n_lines):

        assert oracle_line['sentence'] == bootleg_line['sentence']

        # oracle
        entity2types_oracle = defaultdict(str)
        for entity, type in zip(oracle_line['aliases'], oracle_line['thingtalk_types']):
            assert len(type) == 1
            entity2types_oracle[entity.strip()] = type[0]

        # bootleg
        entity2types_bootleg = defaultdict(list)
        for alias, all_qids, all_probs, span in zip(
            bootleg_line['aliases'], bootleg_line['cands'], bootleg_line['cand_probs'], bootleg_line['spans']
        ):
            type_ids, type_probs, qids = bootleg.collect_features_per_alias(alias, all_probs, all_qids)

            type_vocabs = [bootleg.typeqid_to_type_vocab[bootleg.id2typeqid[id_]] for id_ in type_ids]

            entity2types_bootleg[alias] = type_vocabs

        for oracle_entity, oracle_type in entity2types_oracle.items():
            detected = False
            for bootleg_entity in entity2types_bootleg.keys():
                if oracle_entity in bootleg_entity or bootleg_entity in oracle_entity:
                    # consider it detected as long as it overlaps in some way
                    detected = True

                    correct_type = oracle_type in entity2types_bootleg[bootleg_entity]

                    soft_true_positive[oracle_type] += 1
                    for bootleg_type in entity2types_bootleg[bootleg_entity]:
                        precision_true_positive[bootleg_type] += 1
                    if correct_type:
                        strict_true_positive[oracle_type] += 1
                    else:
                        for bootleg_type in entity2types_bootleg[bootleg_entity]:
                            mismatched_types[oracle_type][bootleg_type] += 1
                    break

            if not detected:
                false_negative[oracle_type] += 1

        for bootleg_entity, bootleg_types in entity2types_bootleg.items():
            is_true_entity = False
            for oracle_entity in entity2types_oracle.keys():
                if oracle_entity in bootleg_entity or bootleg_entity in oracle_entity:
                    # consider it a true entity as long as it overlaps in some way
                    is_true_entity = True
                    break

            if not is_true_entity:
                for bootleg_type in bootleg_types:
                    false_positive[bootleg_type] += 1
                    mispredicted_entities[bootleg_type][bootleg_entity] += 1

    keys = list(set(strict_true_positive.keys()) | set(soft_true_positive.keys()) | \
        set(false_negative.keys()))
    keys.sort()

    for k in keys:
        if precision_true_positive[k] + false_positive[k] != 0:
            precision = precision_true_positive[k] / (precision_true_positive[k] + false_positive[k])
        else:
            precision = 'n/a'
        if strict_true_positive[k] + false_negative[k] != 0:
            strict_recall = strict_true_positive[k] / (strict_true_positive[k] + false_negative[k])
        else:
            strict_recall = 'n/a'
        if soft_true_positive[k] + false_negative[k] != 0:
            soft_recall = soft_true_positive[k] / (soft_true_positive[k] + false_negative[k])
        else:
            soft_recall = 'n/a'

        print(k, precision, strict_recall, soft_recall, sep='\t')

    print()
    for k in keys:
        print(k, ':')
        print('mispredicted types')
        for bootleg_k in mismatched_types[k].most_common(5):
            print(bootleg_k)
        print('unnecessary entities')
        for bootleg_k in mispredicted_entities[k].most_common(5):
            print(bootleg_k)
        print()
