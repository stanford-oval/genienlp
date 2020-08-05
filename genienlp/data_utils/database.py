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

from pytrie import SortedStringTrie as Trie
# import pygtrie

DOMAIN_TYPE_MAPPING = dict()
DOMAIN_TYPE_MAPPING['music'] = {'Person': 'song_artist', 'MusicRecording': 'song_name', 'MusicAlbum': 'song_album'}
DOMAIN_TYPE_MAPPING['spotify'] = {'id': 'song_name', 'song': 'song_name', 'artist': 'song_artist', 'artists': 'song_artist', 'album': 'song_album', 'genres': 'song_genre'}

# Order of types should not be changed (new types can be appended)
TYPES = ('song_name', 'song_artist', 'song_album', 'song_genre')

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

BANNED_WORDS = stopwords.words('english') + ['music', 'name', 'want', 'album', 'please', 'who', 'show me',
                                             'play', 'track', 'song', 'record', 'album', 'something', 'resume',
                                             'find me', 'the', 'search for me', 'search', 'yes', 'yeah']

def is_special_case(i, tokens, key_tokenized):
    if ' '.join(key_tokenized) in BANNED_WORDS:
        return True
    return False


class Database(object):
    def __init__(self, items):
        self.data = Trie(items)
        self.unk_type = 'unk'
        self.type2id = {self.unk_type:  0}
        self.type2id.update({type: i + 1 for i, type in enumerate(TYPES)})
    
    def update_items(self, new_items, allow_new_types=False):
        new_items_processed = dict()
        for token, type in new_items.items():
            if type in self.type2id.keys():
                new_items_processed[token] = type
            elif allow_new_types:
                new_items_processed[token] = type
                self.type2id[type] = len(self.type2id)
            else:
                # type is unknown
                new_items_processed[token] = 'unk'
        
        self.data = Trie(new_items_processed)
    
    def lookup(self, tokens, subset=None, retrieve_method='database'):
        tokens_type_ids = []
        i = 0
        
        if retrieve_method != 'database' and subset is not None:
            if retrieve_method == 'thingtalk':
                # types are retrieved from the program
                lookup_dict = Trie(subset)
            elif retrieve_method == 'answer':
                # prune db (types are retrieved from the database)
                lookup_dict = dict()
                for token, type in self.data.items():
                    if token in subset:
                        lookup_dict[token] = type
        else:
            lookup_dict = self.data
        
        while i < len(tokens):
            token = tokens[i]
            # sort by number of tokens so longer keys get matched first
            matched_items = sorted(lookup_dict.items(prefix=token), key=lambda item: len(item[0]), reverse=True)
            found = False
            for key, type in matched_items:
                key_tokenized = key.split()
                cur = i
                j = 0
                while cur < len(tokens) and j < len(key_tokenized):
                    if tokens[cur] != key_tokenized[j]:
                        break
                    j += 1
                    cur += 1
                
                if j == len(key_tokenized):
                    if is_special_case(i, tokens, key_tokenized):
                        continue
                    
                    # match found
                    found = True
                    tokens_type_ids.extend([self.type2id[type] for _ in range(i, cur)])
                    
                    # move i to current unprocessed position
                    i = cur
                    break
            
            if not found:
                tokens_type_ids.append(self.type2id['unk'])
                i += 1
        return tokens_type_ids
