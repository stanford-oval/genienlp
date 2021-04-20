import argparse
import os
from collections import Counter

import jsonlines
import ujson
from termcolor import colored

from genienlp.data_utils.database_utils import is_banned

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--database_dir', type=str)
parser.add_argument('--threshold', type=int, default=0.1)
parser.add_argument('--domain', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    
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
                
                # # pizzeria
                # if qid == 'Q177':
                #     type = 'Q571'
                #
                # # ramen
                # if qid == 'Q1051265':
                #     type = 'Q1778821'
                
                if 'cuisine' in title or 'pasta' in title or 'culture of ' in title or \
                        title in ['food', 'type of food or dish', 'dish', 'convenience food', 'rice dish',
                                  'dish', 'food ingredient', 'stuffed pasta', 'raw fish dish', 'seafood dish',
                                  'soup', 'country', 'sovereign state', 'noodle', 'intangible cultural heritage']:
                    typeqid = 'Q1778821'
                
                elif title in ['city of the United States', 'big city', 'city with millions of inhabitants',
                               'commune of France', 'city', 'human settlement']:
                    typeqid = 'Q2221906'
                
                elif 'restaurant chain' in title or title in ['restaurant', 'food manufacturer', 'business',
                                                              'enterprise', 'pizzeria chain']:
                    typeqid = 'Q11707'
                
                elif 'writer' in title or title in ['journalist', 'author', 'politician', 'novelist',
                                                    'university teacher', 'Esperantist', 'philosopher', 'actor',
                                                    'composer', 'film actor',
                                                    'painter', 'historian', 'lawyer', 'poet', 'singer', 'singer',
                                                    'musician', 'songwriter', 'composer', 'producer',
                                                    'singer-songwriter', 'musical group', 'drummer',
                                                    'writer', 'philanthropist', 'public figure',
                                                    'poet', 'guitarist', 'rapper', 'painter',
                                                    'film director', 'dancer', 'screenwriter',
                                                    'television presenter', 'film producer',
                                                    'saxophonist', 'music pedagogue',
                                                    'association football player', 'film score composer',
                                                    'disc jockey', 'record producer', 'engineer', 'entrepreneur',
                                                    'boy band', 'musical ensemble', 'artist',
                                                    'vocal group', 'heavy metal band',
                                                    'literary character', 'lawyer', 'lyricist',
                                                    'baseball player', 'pianist', 'recording artist',
                                                    'autobiographer', 'fashion designer']:
                    typeqid = 'Q5'
                
                elif 'postal code' in title or title in ['administrative territorial entity identifier',
                                                         'nominal number', 'unique identifier']:
                    typeqid = 'Q37447'
                
                ########################################################################
                ########################################################################
                
                all_new_types[typeqid] += 1
                
                print(f'{alias}, {prob}, {qid}, {typeqid}: {title}')
            
            else:
                print(f'{alias}, {prob}, {qid}, {typeqid}: ?')
    
    with open(os.path.join(args.database_dir, 'es_material/almond_type2qid.json'), 'r') as fin:
        almond_type2qid = ujson.load(fin)
    
    print(f'all_titles: {all_titles.most_common()}')
    main_types = []
    extra_types = []
    
    values = almond_type2qid[args.domain].values()
    
    for tup in all_new_types.most_common():
        if tup[0] in values:
            main_types.append(tup)
        else:
            extra_types.append(tup)
    
    print('all_new_types:', *[colored(tup, "red") for tup in main_types], *[extra_types])
