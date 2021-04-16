import argparse
import os
import ujson
from collections import defaultdict
from pprint import pprint

MAPPING = defaultdict(dict)

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--bootleg_input_dir', type=str)
parser.add_argument('--threshold', type=int, default=0.5)

args = parser.parse_args()


MAPPING['spotify'] = {'song': ['song', 'single', 'musical composition', 'ballad', 'extended play', 'literary work',
                         'television series', 'film', 'play'],
                    'album': ['album'],
                    'genre': ['genre', 'country', 'music by country or region', 'music term', 'republic',
                                               'ethnic group', 'music scene', 'music style'],
                    'artist': ['person', 'musician', 'singer', 'actor', 'musician', 'songwriter',
                              'composer', 'singer-songwriter', 'musical group', 'drummer',
                              'writer', 'poet', 'guitarist', 'rapper', 'painter',
                              'film director', 'rock band', 'university teacher', 'journalist',
                              'television presenter', 'saxophonist', 'music pedagogue',
                              'association football player', 'disc jockey', 'record producer', 'engineer',
                              'human biblical figure', 'big band', 'musical duo', 'girl group',
                              'boy band', 'musical ensemble', 'artist', 'vocal group', 'heavy metal band',
                              'literary character', 'lawyer', 'lyricist', 'baseball player']}

MAPPING['restaurants'] = {'Restaurant': ['@restaurant chain@', 'restaurant', 'food manufacturer'],
                            'Person': ['@writer@'],
                            'servesCuisine': ['@cuisine@', '@pasta@', '@culture of@',
                                              'food', 'type of food or dish', 'dish', 'convenience food', 'rice dish',
                                          'dish', 'food ingredient', 'stuffed pasta', 'raw fish dish',
                                          'soup', 'country', 'sovereign state', 'noodle', 'intangible cultural heritage'],
                            'Location': ['city of the United States', 'big city', 'city with millions of inhabitants',
                                       'commune of France']}

MAPPING['movies'] = {'Movie': ['@film@', 'song', 'single', 'media franchise', 'literary work', 'television series', 'written work'],
                            'creator': ['@producer@'],
                            'director': ['@director@'],
                            'actor': ['@actor@', '@actress@'],
                            'genre': ['@genre@', '@fiction@', 'drama', 'comedy'],
                             'iso_lang_code': ['@language@', 'cinema of country or region']}


MAPPING['music'] = {'MusicRecording': ['song', 'single', 'musical composition', 'ballad', 'extended play', 'literary work',
                         'television series', 'film', 'play'],
                            'MusicAlbum': ['@album@'],
                            'Person': ['@person@', '@actor@', '@musician',
                                       'singer', 'musician', 'songwriter',
                                      'composer', 'producer',
                                      'singer-songwriter', 'musical group', 'drummer',
                                      'writer', 'philanthropist', 'public figure',
                                      'poet', 'guitarist', 'rapper', 'painter',
                                      'film director', 'dancer', 'screenwriter',
                                      'rock band', 'university teacher', 'journalist',
                                      'television presenter', 'film producer',
                                      'saxophonist', 'music pedagogue',
                                      'association football player', 'film score composer',
                                      'disc jockey', 'record producer', 'engineer',
                                      'human biblical figure', 'big band',
                                      'musical duo', 'girl group', 'entrepreneur',
                                      'boy band', 'musical ensemble', 'artist',
                                      'vocal group', 'heavy metal band',
                                      'literary character', 'lawyer', 'lyricist',
                                      'baseball player', 'pianist', 'recording artist',
                                      'autobiographer', 'fashion designer'],
                            'genres': ['@genre@', 'country', 'music by country or region', 'music term', 'republic',
                                               'ethnic group', 'music scene', 'popular music', 'rock music',
                                               'heavy metal', 'music', 'pop music', 'electronic music', 'music style'],
                             'iso_lang_code': ['@language@', 'cinema of country or region', 'sovereign state', 'Bantu',
                                                  'Serbo-Croatian', 'big city', 'Upper Guinea Creoles']}

MAPPING['books'] = {'Book': ['@book@', '@novel@', '@poem@',
                             'written work', 'literary work', 'literature', 'play', 'film', 'occurrence', 'song',
                             'fictional human', 'profession',
                             'document', 'day of the week', 'compilation album', 'magazine', 'television series', 'taxon',
                             'Bible translation',
                             'concept', 'disease', 'technique', 'activity', 'food', 'political ideology', 'literary genre',
                             'mountain', 'mental process',
                             'academic discipline', 'base material', 'negative emotion', 'emotion'],
                    'Person': ['@person@', '@rights activist@', '@writer@',
                               'journalist', 'author', 'politician', 'Esperantist', 'philosopher', 'actor',
                                  'painter', 'historian', 'lawyer', 'poet', 'singer'],
                    'award': ['@award@', 'recurring event'],
                    'bookEdition': ['@publisher@', 'editorial collection', 'version, edition, or translation']}

MAPPING['thingpedia'] = {
                        ('artist', 'person'): ['@artist@', '@musician@', '@composer@',
                            'singer', 'actor', 'musician', 'songwriter', 'pianist', 'jazz guitarist',
                              'composer', 'singer-songwriter', 'musical group', 'drummer', 'keyboardist',
                              'poet', 'guitarist', 'rapper', 'rock band', 'saxophonist', 'music pedagogue',
                              'opera singer', 'disc jockey', 'record producer', 'big band', 'musical duo', 'girl group',
                              'music arranger', 'boy band', 'musical ensemble', 'artist', 'lyricist', 'bandleader',
                              'bassist', 'banjoist',
            
                              # debatable mapping
                              'film actor', 'television actor', 'writer', 'conductor', 'stage actor', 'engineer',
                              'model', 'university teacher', 'human biblical figure', 'literary character', 'lawyer',
                              'lyricist', 'baseball player', 'dancer', 'screenwriter', 'voice actor', 'film producer',
                              'politician', 'film director', 'producer', 'painter'],
                          
                          'album': ['@album@'],
                          
                          'genres': ['@genre@',
                                     'country', 'music by country or region', 'music term', 'republic',
                                   'ethnic group', 'music scene', 'music style', 'rock music',
                                   'heavy metal', 'pop music', 'hip hop music', 'electronic music'],
    
                            'song': ['song', 'musical composition', 'ballad', 'extended play', 'single',
    
                                       # debatable mapping
                                       'literary work', 'television series', 'film', 'play'],
    
                            'restaurant_cuisine': ['@cuisine@', '@dish@', '@pasta@',
                                                   'food'],
    
                            'restaurant': ['@restaurant chain@',
                                            'restaurant'],
    
                            'Location': ['@city@',
                                            'location', 'lake', 'sovereign state'],
    
                            'device': ['@device@'],
    
                            'iot_name': ['@room@'],
                            'dog': ['dog breed'],
    
                          }



            
pprint(MAPPING)
with open(os.path.join(args.bootleg_input_dir, f'es_material/almond_type_mapping.json'), 'w') as fout:
    ujson.dump(MAPPING, fout)

NEW_MAPPING_match = defaultdict(dict)
NEW_MAPPING_include = defaultdict(dict)
for domain, mapping in MAPPING.items():
    title2TTtype_match = {}
    title2TTtype_include = {}
    for TTtype, titles in mapping.items():
        for title in titles:
            if title[0] == title[-1] == '@':
                title2TTtype_include[title.strip('@')] = TTtype
            else:
                title2TTtype_match[title] = TTtype
    NEW_MAPPING_match[domain] = title2TTtype_match
    NEW_MAPPING_include[domain] = title2TTtype_include
    
with open(os.path.join(args.bootleg_input_dir, f'es_material/almond_type_mapping_matching.json'), 'w') as fout:
    ujson.dump(NEW_MAPPING_match, fout)
with open(os.path.join(args.bootleg_input_dir, f'es_material/almond_type_mapping_inclusion.json'), 'w') as fout:
    ujson.dump(NEW_MAPPING_include, fout)
    
    
    
            
    
    
