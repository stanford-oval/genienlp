from collections import Counter, defaultdict

import jsonlines

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


def main(args):

    # args.ned_normalize_types = 'soft'
    # args.bootleg_prob_threshold = 0.3
    # args.max_types_per_qid = 1
    # args.max_qids_per_entity = 1

    args.ned_normalize_types = 'strict'
    args.bootleg_prob_threshold = 0.01
    args.max_types_per_qid = 2
    args.max_qids_per_entity = 2

    args.max_features_size = args.max_types_per_qid * args.max_qids_per_entity

    bootleg = BatchBootlegEntityDisambiguator(args)

    bootleg_lines = jsonlines.open(args.bootleg_labels, 'r')
    oracle_lines = jsonlines.open(args.oracle_labels, 'r')

    not_detected = Counter()
    no_TT_type = Counter()
    not_correct = Counter()

    fout = open(args.output_file, 'w')

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
            type_ids, type_probs, qids = bootleg.collect_features_per_alias(alias, all_probs, all_qids)

            type_vocabs = [bootleg.typeqid_to_type_vocab[bootleg.id2typeqid[id_]] for id_ in type_ids]

            entity2types_bootleg[alias] = type_vocabs

        for k_oracle, v_oracle in entity2types_oracle.items():
            detected = False
            for k_boot in entity2types_bootleg.keys():
                if k_oracle in k_boot or k_boot in k_oracle:
                    detected = True
                    break

            if detected:
                if len(entity2types_bootleg[k_boot]) == 0:
                    no_TT_type[k_oracle] += 1
                elif v_oracle not in entity2types_bootleg[k_boot]:
                    not_correct[k_oracle] += 1
                    fout.write(
                        f'TT type for {k_oracle}: {entity2types_oracle[k_oracle]} ; {k_boot}: {entity2types_bootleg[k_boot]}'
                        + '\n'
                    )
            else:
                not_detected[k_oracle] += 1

    fout.close()

    print(f'not_detected: {not_detected}')
    print(f'no TT type detected: {no_TT_type}')
    print(f'not_correct: {not_correct}')

    all_wrong = sum(not_detected.values()) + sum(no_TT_type.values()) + sum(not_correct.values())
    print(f'not_detected: {sum(not_detected.values()) / all_wrong}')
    print(f'no TT type detected: {sum(no_TT_type.values()) / all_wrong}')
    print(f'not_correct: {sum(not_correct.values()) / all_wrong}')
