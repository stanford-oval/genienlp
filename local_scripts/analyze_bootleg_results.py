import argparse
import os
import ujson
import jsonlines
from genienlp.data_utils.database_utils import is_banned
from termcolor import colored
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--database_dir', type=str)
parser.add_argument('--threshold', type=int, default=0.1)

args = parser.parse_args()

with open(f'{args.database_dir}/wiki_entity_data/type_mappings/wiki/qid2typenames.json', 'r') as fin:
    qid2typenames = ujson.load(fin)
with open(f'{args.database_dir}/wiki_entity_data/type_mappings/wiki/type_vocab_to_wikidataqid.json', 'r') as fin:
    type_vocab_to_wikidataqid = ujson.load(fin)
    wikidataqid_to_type_vocab = {v: k for k, v in type_vocab_to_wikidataqid.items()}
with open(f'{args.database_dir}/es_material/typeqid2id.json', 'r') as fin:
    typeqid2id = ujson.load(fin)

all_types = []
all_aliases = []
all_qids = []
all_probs = []
unknown_qids = set()

lines = jsonlines.open(args.input_file, 'r')

for line in lines:
    qids = line['qids']
    aliases = line['aliases']
    probs = line['probs']
    for alias, qid, prob in zip(aliases, qids, probs):
        if prob < args.threshold or is_banned(alias):
            continue
        
        types = []
        if qid in qid2typenames and qid2typenames[qid]:
            # map entity qid to its type titles on wikidata ; then map titles to their wikidata qids
            for typename in qid2typenames[qid]:
                if typename in type_vocab_to_wikidataqid:
                    types.append(type_vocab_to_wikidataqid[typename])
        
        all_aliases.append(alias)
        all_qids.append(qid)
        all_types.append(types)
        all_probs.append(prob)

assert len(all_qids) == len(all_aliases) == len(all_types) == len(all_probs)

all_titles = Counter()
all_new_types = Counter()
for qid, alias, types, prob in zip(all_qids, all_aliases, all_types, all_probs):
    # assert type in type2id
    for typeqid in types:
        if typeqid in wikidataqid_to_type_vocab:
            title = wikidataqid_to_type_vocab[typeqid]
            all_titles[title] += 1
            
            print(f'{alias}, {prob}, {qid}, {typeqid}: {title}')
            
            ######
            ## copy this code block to database_utils.post_process_bootleg_types when done with your analysis
            ######
            ########################################################################
            ########################################################################
            
            ########################################################################
            ########################################################################
            
            all_new_types[typeqid] += 1
            
            print(f'{alias}, {prob}, {qid}, {typeqid}: {title}')
        
        else:
            print(f'{alias}, {prob}, {qid}, {typeqid}: ?')

# DOMAIN = 'cross_ner/news'
DOMAIN = 'thingpedia'

with open(os.path.join(args.database_dir, 'es_material/almond_type2qid.json'), 'r') as fin:
    almond_type2qid = ujson.load(fin)

print(f'all_titles: {all_titles.most_common()}')
main_types = []
extra_types = []

for tup in all_new_types.most_common():
    if tup[0] in almond_type2qid[DOMAIN].values():
        main_types.append(tup)
    else:
        extra_types.append(tup)

print('all_new_types:', *[colored(tup, "red") for tup in main_types], *[extra_types])
