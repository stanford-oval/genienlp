import argparse
import os
import ujson
import jsonlines
from genienlp.data_utils.database_utils import BANNED_PHRASES, BANNED_REGEXES, DOMAIN_TYPE_MAPPING
from termcolor import colored
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--bootleg_input_dir', type=str)
parser.add_argument('--threshold', type=int, default=0.5)

args = parser.parse_args()


wikitype2almondtype = {}

with open(os.path.join(args.bootleg_input_dir, 'emb_data/entityQID_to_wikidataTypeQID.json'), 'r') as fin:
    qid2type = ujson.load(fin)

# with open(os.path.join(args.bootleg_input_dir, 'emb_data/es_qid2type.json'), 'r') as fin:
#     qid2type = ujson.load(fin)
    
with open(os.path.join(args.bootleg_input_dir, 'emb_data/wikidatatitle_to_typeid_0905.json'), 'r') as fin:
    title2typeid = ujson.load(fin)
    typeid2title = {v: k for k, v in title2typeid.items()}
    
# with open(os.path.join(args.bootleg_input_dir, 'emb_data/wikidataqid_to_bootlegtypeid.json'), 'r') as fin:
#     type2id = ujson.load(fin)
    
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
        if prob < args.threshold or alias in BANNED_PHRASES or any([regex.match(alias) for regex in BANNED_REGEXES]):
            continue
        
        if qid in qid2type:
            if isinstance(qid2type[qid], list):
                if len(qid2type[qid]):
                    all_types.append(qid2type[qid])
                else:
                    all_types.append('unk')
                    unknown_qids.add(qid)
            else:
                all_types.append(qid2type[qid])
        else:
            all_types.append('unk')
            unknown_qids.add(qid)
            
        all_aliases.append(alias)
        all_qids.append(qid)
        all_probs.append(prob)
        
assert len(all_qids) == len(all_aliases) == len(all_types)

all_titles = Counter()
all_new_types = Counter()
for qid, alias, types, prob in zip(all_qids, all_aliases, all_types, all_probs):
    # assert type in type2id
    for type in types:
        if type in typeid2title:
            title = typeid2title[type]
            all_titles[title] += 1
    
            print(f'{alias}, {prob}, {qid}, {type}: {title}')
            
            ######
            ## copy this code block to database_utils.post_process_bootleg_types when done with your analysis
            ######
            ########################################################################
            ########################################################################

            # rap, rap music
            if qid in ['Q6010', 'Q11401']:
                type = 'Q188451'

            if title in ['song', 'single', 'musical composition', 'ballad', 'extended play', 'literary work',
                         'television series', 'film', 'play']:
                type = 'Q7366'
            elif 'album' in title or title in []:
                type = 'Q482994'
            elif 'genre' in title or title in ['country', 'music by country or region', 'music term', 'republic',
                                               'ethnic group', 'music scene']:
                type = 'Q188451'
            elif 'person' in title or 'musician' in title or \
                    title in ['singer', 'actor', 'musician', 'songwriter',
                              'composer', 'singer-songwriter', 'musical group', 'drummer',
                              'writer', 'poet', 'guitarist', 'rapper', 'painter',
                              'film director', 'rock band', 'university teacher', 'journalist',
                              'television presenter', 'saxophonist', 'music pedagogue',
                              'association football player', 'disc jockey', 'record producer', 'engineer',
                              'human biblical figure', 'big band', 'musical duo', 'girl group',
                              'boy band', 'musical ensemble', 'artist', 'vocal group', 'heavy metal band',
                              'literary character', 'lawyer', 'lyricist', 'baseball player']:
                type = 'Q5'

            elif title in ['video game', 'disease', 'city of the United States', 'taxon',
                           'Wikimedia disambiguation page', 'Wikimedia list article']:
                type = 'unk'
            
            ########################################################################
            ########################################################################
    
            all_new_types[type] += 1
            
            # print(f'{alias}, {prob}, {qid}, {type}: {title}')
    
            
        else:
            print(f'{alias}, {prob}, {qid}, {type}: ?')
        

print(f'all_titles: {all_titles.most_common()}')
print('all_new_types:', *[colored(tup, "red") if tup[0] in DOMAIN_TYPE_MAPPING['restaurants'].values() else tup for tup in all_new_types.most_common()])
