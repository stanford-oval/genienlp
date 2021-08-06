import argparse
from collections import Counter, defaultdict

import jsonlines

from genienlp.ned import BatchBootlegEntityDisambiguator
from genienlp.ned.ned_utils import reverse_bisect_left

parser = argparse.ArgumentParser()

parser.add_argument('--bootleg_labels', type=str)
parser.add_argument('--oracle_labels', type=str)
parser.add_argument('--database_dir', type=str)
parser.add_argument('--ned_domains', type=str, nargs='+')

args = parser.parse_args()


if __name__ == '__main__':

    args.bootleg_model = 'bootleg_uncased_mini'
    args.bootleg_post_process_types = True
    args.bootleg_output_dir = 'results_temp'
    args.embeddings = '.embeddings'
    args.almond_type_mapping_path = None
    args.bootleg_prob_threshold = 0.3
    args.max_types_per_qid = 1
    args.max_qids_per_entity = 1
    args.max_features_size = args.max_types_per_qid * args.max_qids_per_entity

    bootleg = BatchBootlegEntityDisambiguator(args)

    bootleg_lines = jsonlines.open(args.bootleg_labels, 'r')
    oracle_lines = jsonlines.open(args.oracle_labels, 'r')

    not_detected = Counter()
    no_TT_type = Counter()
    not_correct = Counter()

    for oracle_line, bootleg_line in zip(oracle_lines, bootleg_lines):

        assert len(oracle_line['sentence'].split(' ')) == len(bootleg_line['sentence'].split(' '))

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
            # filter qids with confidence lower than a threshold
            idx = reverse_bisect_left(all_probs, args.bootleg_prob_threshold)
            all_qids = all_qids[:idx]
            all_probs = all_probs[:idx]

            type_ids, type_probs, qids = bootleg.collect_features_per_alias(alias, all_probs, all_qids)

            type_vocabs = [bootleg.typeqid_to_type_vocab[bootleg.id2typeqid[id_]] for id_ in type_ids]

            entity2types_bootleg[alias] = type_vocabs

        for k, v in entity2types_oracle.items():
            if k not in entity2types_bootleg:
                if 'the ' + k not in entity2types_bootleg:
                    not_detected[k] += 1
                else:
                    k = 'the ' + k
            elif len(entity2types_bootleg[k]) == 0:
                no_TT_type[k] += 1
            elif v not in entity2types_bootleg[k]:
                not_correct[k] += 1
                # print(f'correct TT type: {entity2types_oracle[k]}')
                # print(f'bootleg TT type: {entity2types_bootleg[k]}')

    print(f'not_detected: {not_detected}')
    print(f'no TT type detected: {no_TT_type}')
    print(f'not_correct: {not_correct}')

    all_wrong = sum(not_detected.values()) + sum(no_TT_type.values()) + sum(not_correct.values())
    print(f'not_detected: {sum(not_detected.values()) / all_wrong}')
    print(f'no TT type detected: {sum(no_TT_type.values()) / all_wrong}')
    print(f'not_correct: {sum(not_correct.values()) / all_wrong}')
