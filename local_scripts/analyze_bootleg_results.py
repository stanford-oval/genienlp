import argparse
import os
from collections import Counter

import jsonlines
import ujson
from termcolor import colored

from genienlp.data_utils.bootleg import Bootleg
from genienlp.data_utils.database_utils import is_banned

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--database_dir', type=str)
parser.add_argument('--threshold', type=int, default=0.1)
parser.add_argument('--almond_domains', type=str, nargs='+')
parser.add_argument('--subsample', type=int, default='1000000000')


args = parser.parse_args()

if __name__ == '__main__':

    args.bootleg_model = 'bootleg_uncased_mini'
    args.bootleg_post_process_types = True
    args.bootleg_output_dir = 'results_temp'
    args.embeddings = ' embeddings'

    bootleg = Bootleg(args)

    all_typeqids = []
    all_aliases = []
    all_qids = []
    all_probs = []
    unknown_qids = set()

    lines = jsonlines.open(args.input_file, 'r')

    for count, line in enumerate(lines):
        if count >= args.subsample:
            break
        qids = line['qids']
        aliases = line['aliases']
        probs = line['probs']
        for alias, qid, prob in zip(aliases, qids, probs):
            if prob < args.threshold or is_banned(alias):
                continue

            typeqids = []
            if qid in bootleg.qid2typenames and bootleg.qid2typenames[qid]:
                # map entity qid to its type titles on wikidata ; then map titles to their wikidata qids
                for typename in bootleg.qid2typenames[qid]:
                    if typename in bootleg.type_vocab_to_entityqid:
                        typeqids.append(bootleg.type_vocab_to_entityqid[typename])

            all_aliases.append(alias)
            all_qids.append(qid)
            all_typeqids.append(typeqids)
            all_probs.append(prob)

    assert len(all_qids) == len(all_aliases) == len(all_typeqids) == len(all_probs)

    all_titles = Counter()
    all_new_types = Counter()
    for qid, alias, types, prob in zip(all_qids, all_aliases, all_typeqids, all_probs):
        # assert type in type2id
        for typeqid in types:
            if typeqid in bootleg.entityqid_to_type_vocab:
                title = bootleg.entityqid_to_type_vocab[typeqid]
                all_titles[title] += 1

                print(f'{alias}, {prob:.3f}, {qid}, {typeqid}: {title}')

                ######
                ## copy this code block to database_utils.post_process_bootleg_types when done with your analysis
                ######
                ########################################################################
                ########################################################################

                # do type mapping

                if True:
                    # map qid to title
                    title = bootleg.entityqid_to_type_vocab[typeqid]
                    # process may return multiple types for a single type when it's ambiguous
                    mapped_typeqid = bootleg.post_process_bootleg_types(title)

                    # attempt to normalize qids failed; just use the original type
                    if mapped_typeqid is not None:
                        typeqid = mapped_typeqid

                ########################################################################
                ########################################################################
                if isinstance(typeqid, str):
                    all_new_types[typeqid] += 1
                else:
                    for item in typeqid:
                        all_new_types[item] += 1

                print(f'{alias}, {prob}, {qid}, {typeqid}: {title}')

            else:
                print(f'{alias}, {prob}, {qid}, {typeqid}: ?')

    with open(os.path.join(args.database_dir, 'es_material/almond_type2qid.json'), 'r') as fin:
        almond_type2qid = ujson.load(fin)

    print(f'all_titles: {all_titles.most_common()}')
    main_types = []
    extra_types = []

    for tup in all_new_types.most_common():
        if bootleg.entityqid_to_type_vocab[tup[0]] in bootleg.almond_type_mapping:
            main_types.append(tup)
        else:
            extra_types.append(tup)

    print('all_new_types:', *[colored(tup, "red") for tup in main_types], *extra_types)
