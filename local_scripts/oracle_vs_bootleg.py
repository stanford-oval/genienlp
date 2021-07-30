import argparse
from collections import Counter, defaultdict

import jsonlines
import ujson

from genienlp import run_bootleg
from genienlp.data_utils.bootleg import post_process_bootleg_types, reverse_bisect_left
from genienlp.data_utils.database_utils import DOMAIN_TYPE_MAPPING, is_banned

parser = argparse.ArgumentParser()

parser.add_argument('--bootleg_labels', type=str)
parser.add_argument('--oracle_labels', type=str)

run_bootleg.parse_argv(parser)

args = parser.parse_args()


bootleg_lines = jsonlines.open(args.bootleg_labels, 'r')
oracle_lines = jsonlines.open(args.oracle_labels, 'r')

with open(f'{args.database_dir}/emb_data/entityQID_to_wikidataTypeQID.json', 'r') as fin:
    qid2type = ujson.load(fin)
with open(f'{args.database_dir}/es_material/type2id.json', 'r') as fin:
    type2id = ujson.load(fin)

with open(f'{args.database_dir}/emb_data/wikidatatitle_to_typeid_0905.json', 'r') as fin:
    title2typeid = ujson.load(fin)
    typeid2title = {v: k for k, v in title2typeid.items()}

TTtype2DBtype = dict()
for domain in args.almond_domains:
    TTtype2DBtype.update(DOMAIN_TYPE_MAPPING[domain])
DBtype2TTtype = {v: k for k, v in TTtype2DBtype.items()}


not_detected = Counter()
no_TT_type = Counter()
not_correct = Counter()

for oracle_line, bootleg_line in zip(oracle_lines, bootleg_lines):

    assert len(oracle_line['sentence'].split(' ')) == len(bootleg_line['sentence'].split(' '))

    entity2types_oracle = defaultdict(str)
    # oracle
    for entity, type in zip(oracle_line['aliases'], oracle_line['thingtalk_types']):
        assert len(type) == 1
        entity2types_oracle[entity.strip()] = type[0]

    # bootleg

    entity2types_bootleg = defaultdict(list)
    tokenized = bootleg_line['sentence'].split(' ')
    for alias, all_qids, all_probs, span in zip(
        bootleg_line['aliases'], bootleg_line['cands'], bootleg_line['cand_probs'], bootleg_line['spans']
    ):
        # filter qids with confidence lower than a threshold
        idx = reverse_bisect_left(all_probs, args.bootleg_prob_threshold)
        all_qids = all_qids[:idx]
        all_probs = all_probs[:idx]

        TTtypes = []
        if not is_banned(alias):
            for qid, prob in zip(all_qids, all_probs):
                # get all type for a qid
                if qid in qid2type:
                    all_types = qid2type[qid]
                else:
                    all_types = []

                if isinstance(all_types, str):
                    all_types = [all_types]

                if len(all_types):
                    # update
                    # go through all types
                    for type in all_types:
                        if type in type2id:
                            title = typeid2title.get(type, '?')

                            ## map wikidata types to thingtalk types
                            if args.bootleg_post_process_types:
                                type = post_process_bootleg_types(qid, type, title, args.almond_domains)

                            if type in DBtype2TTtype:
                                TTtypes.append(DBtype2TTtype[type])

            entity2types_bootleg[alias] = TTtypes

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
