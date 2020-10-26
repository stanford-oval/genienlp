#
# Copyright (c) 2019, The Board of Trustees of the Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
import unicodedata
import re

import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords

tracer = logging.getLogger('elasticsearch')
tracer.setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

DOMAIN_TYPE_MAPPING = dict()

## SQA
DOMAIN_TYPE_MAPPING['music'] = {'Person': 'Q5', 'MusicRecording': 'Q7366', 'MusicAlbum': 'Q208569'}   # Q5:human, Q7366:song, Q208569: studio album

DOMAIN_TYPE_MAPPING['movies'] = {'Person': 'Q5', 'Movie': 'Q11424'}   # Q11424:film

DOMAIN_TYPE_MAPPING['books'] = {'Person': 'Q5', 'Book': 'Q571'}   # Q571:book

DOMAIN_TYPE_MAPPING['linkedin'] = {'Person': 'Q5', 'Organization': 'Q43229', 'address.addressLocality': 'Q319608', 'award': 'Q618779'} # Q319608:postal address

DOMAIN_TYPE_MAPPING['restaurants'] = {'Person': 'Q5', 'Restaurant': 'Q571', 'servesCuisine': 'Q1778821', 'geo': 'Q2221906',
                                      'address.postalCode': 'Q37447', 'aggregateRating.ratingValue': 'Q2283373',
                                      'aggregateRating.reviewCount': 'Q265158'}   # Q2221906:geographic location, Q2283373: restaurant rating, Q265158: review

DOMAIN_TYPE_MAPPING['hotels'] = {'Hotel': 'Q571', 'LocationFeatureSpecification': 'Q5912147', 'geo': 'Q2221906',
                                 'CheckinTime': 'Q1068755', 'CheckoutTime': 'Q56353377', 'starRating.ratingValue': 'Q2976556'}   # Q5912147:hotel amenity, Q2976556:hotel rating

# Dialogues

DOMAIN_TYPE_MAPPING['spotify'] = {'id': 'Q134556', 'song': 'Q7366', 'artist': 'Q5',
                                  'artists': 'Q5', 'album': 'Q208569', 'genres': 'Q188451'}   # Q188451:music genre

# Order of types should not be changed (new types can be appended)
TYPES = ('song_name', 'song_artist', 'song_album', 'song_genre')


BANNED_WORDS = set(
    stopwords.words('english') + \
    ['music', 'musics', 'name', 'names', 'want', 'wants', 'album', 'albums', 'please', 'who', 'show me',
     'play', 'play me', 'plays', 'track', 'tracks', 'song', 'songs', 'record', 'records', 'recordings', 'album',
     'something', 'get', 'selections',
     'resume', 'resumes', 'find me', 'the', 'search for me', 'search', 'searches', 'yes', 'yeah', 'popular',
     'release', 'released', 'dance', 'dancing', 'need', 'i need', 'i would', ' i will', 'find', 'the list', 'get some']
)

def is_special_case(key):
    if key in BANNED_WORDS:
        return True
    return False

def normalize_text(text):
    text = unicodedata.normalize('NFD', text).lower()
    text = re.sub('\s\s+', ' ', text)
    return text

def has_overlap(start, end, used_aliases):
    for alias in used_aliases:
        alias_start, alias_end = alias[1], alias[2]
        if start < alias_end and end > alias_start:
            return True
    return False

class Database(object):
    def __init__(self, canonical2type, type2id, all_canonicals, TTtype2DBtype):
        self.canonical2type = canonical2type
        self.type2id = type2id
        self.all_canonicals = all_canonicals
        self.entity_type_white_list = list(TTtype2DBtype.values())

        self.unk_type = 'unk'
        self.unk_id = self.type2id[self.unk_type]

    def lookup_ngrams(self, tokens, min_entity_len, max_entity_len):
    
        tokens_type_ids = [self.unk_id] * len(tokens)
    
        max_entity_len = min(max_entity_len, len(tokens))
        min_entity_len = min(min_entity_len, len(tokens))

        pos_tagged = nltk.pos_tag(tokens)
        verbs = set([x[0] for x in pos_tagged if x[1].startswith('V')])
        
        used_aliases = []
        for n in range(max_entity_len, min_entity_len-1, -1):
            ngrams = nltk.ngrams(tokens, n)
            start = -1
            end = n - 1
            for gram in ngrams:
                start += 1
                end += 1
                gram_text = normalize_text(" ".join(gram))
            
                if not is_special_case(gram_text) and not gram_text in verbs and gram_text in self.all_canonicals:
                    if has_overlap(start, end, used_aliases):
                        continue
                    if self.canonical2type[gram_text] not in self.entity_type_white_list:
                        continue
                
                    used_aliases.append(([self.type2id.get(self.canonical2type[gram_text], self.unk_id), start, end]))
    
        for type_id, beg, end in used_aliases:
            tokens_type_ids[beg:end] = [type_id] * (end - beg)
            
        return tokens_type_ids

    def lookup_smaller(self, tokens):
        
        tokens_type_ids = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            # sort by number of tokens so longer keys get matched first
            matched_items = sorted(self.all_canonicals.keys(token), key=lambda item: len(item), reverse=True)
            found = False
            for key in matched_items:
                type = self.canonical2type[key]
                key_tokenized = key.split()
                cur = i
                j = 0
                while cur < len(tokens) and j < len(key_tokenized):
                    if tokens[cur] != key_tokenized[j]:
                        break
                    j += 1
                    cur += 1
                
                if j == len(key_tokenized):
                    if is_special_case(' '.join(key_tokenized)):
                        continue
                    
                    # match found
                    found = True
                    tokens_type_ids.extend([self.type2id[type] for _ in range(i, cur)])
                    
                    # move i to current unprocessed position
                    i = cur
                    break
            
            if not found:
                tokens_type_ids.append(self.unk_id)
                i += 1
        
        return tokens_type_ids
    
    def lookup_longer(self, tokens):
        i = 0
        tokens_type_ids = []
        
        length = len(tokens)
        found = False
        while i < length:
            end = length
            while end > i:
                tokens_str = ' '.join(tokens[i:end])
                if tokens_str in self.all_canonicals:
                    # match found
                    found = True
                    tokens_type_ids.extend([self.type2id[self.canonical2type[tokens_str]] for _ in range(i, end)])
                    # move i to current unprocessed position
                    i = end
                    break
                else:
                    end -= 1
            if not found:
                tokens_type_ids.append(self.unk_id)
                i += 1
            found = False
        
        return tokens_type_ids

    
    def lookup_entities(self, tokens, entities):
        tokens_type_ids = [self.unk_id] * len(tokens)
        tokens_text = " ".join(tokens)
        
        for ent in entities:
            if ent not in self.all_canonicals:
                continue
            ent_num_tokens = len(ent.split(' '))
            idx = tokens_text.index(ent)
            token_pos = len(tokens_text[:idx].strip().split(' '))
            
            type = self.type2id.get(self.canonical2type[ent], self.unk_id)
            
            tokens_type_ids[token_pos: token_pos+ent_num_tokens] = [type]*ent_num_tokens
        
        return tokens_type_ids
        
    
    def lookup(self, tokens, lookup_method=None, min_entity_len=2, max_entity_len=4, answer_entities=None):
        
        if answer_entities is not None:
            tokens_type_ids = self.lookup_entities(tokens, answer_entities)
        
        if lookup_method == 'smaller_first':
            tokens_type_ids = self.lookup_smaller(tokens)
        elif lookup_method == 'longer_first':
            tokens_type_ids = self.lookup_longer(tokens)
        elif lookup_method == 'ngrams':
            tokens_type_ids = self.lookup_ngrams(tokens, min_entity_len, max_entity_len)

        return tokens_type_ids
