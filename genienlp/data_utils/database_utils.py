#
# Copyright (c) 2020-2021 The Board of Trustees of the Leland Stanford Junior University
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


import unicodedata
import re

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


DOMAIN_TYPE_MAPPING = dict()

## SQA


DOMAIN_TYPE_MAPPING['music'] = {'MusicRecording': 'Q7366',
                                'MusicAlbum': 'Q482994',
                                'Person': 'Q5',
                                'iso_lang_code': 'Q315',
                                'genres': 'Q188451'}  # Q188451:music genre, Q5:human, Q7366:song, Q482994:album


# TODO actor and director should be handled differently
DOMAIN_TYPE_MAPPING['movies'] = {'Movie': 'Q11424',
                                 'creator': 'Q2500638',
                                 'director': 'Q3455803',
                                 'actor': 'Q33999',
                                 'genre': 'Q201658',
                                 'iso_lang_code': 'Q315'}  # Q11424:film, Q201658:film genre

# isbn is ignored for books because (1.bootleg doesn't detect them. 2.they are easy to recognize by semantic parser)
DOMAIN_TYPE_MAPPING['books'] = {'Book': 'Q571',
                                'Person': 'Q5',
                                'inLanguage': 'Q315',
                                'iso_lang_code': 'Q315',
                                'award': 'Q618779',
                                'bookEdition': 'Q57933693'}  # Q571:book, Q315:language, Q618779:award

DOMAIN_TYPE_MAPPING['linkedin'] = {'Organization': 'Q43229',
                                   'Person': 'Q5',
                                   'addressLocality': 'Q2221906',
                                   'award': 'Q618779'}   # Q2221906:geographic_location
# linkedin alias
DOMAIN_TYPE_MAPPING['people'] = {'Organization': 'Q43229',
                                   'Person': 'Q5',
                                   'addressLocality': 'Q2221906',
                                   'award': 'Q618779'}   # Q2221906:geographic_location

DOMAIN_TYPE_MAPPING['restaurants'] = {'Restaurant': 'Q571',
                                      'Person': 'Q5',
                                      'servesCuisine': 'Q1778821',
                                      'Location': 'Q2221906',
                                      'postalCode': 'Q37447',
                                      'ratingValue': 'Q2283373',
                                      'reviewCount': 'Q265158'}   # Q2283373:restaurant_rating, Q265158:review, Q1778821:cuisine

DOMAIN_TYPE_MAPPING['hotels'] = {'Hotel': 'Q571',
                                 'LocationFeatureSpecification': 'Q5912147',
                                 'Location': 'Q2221906',
                                 'CheckinTime': 'Q1068755',
                                 'CheckoutTime': 'Q56353377',
                                 'ratingValue': 'Q2976556'}   # Q5912147:hotel_amenity, Q2976556:hotel_rating


## Dialogues
DOMAIN_TYPE_MAPPING['spotify'] = {'song': 'Q7366',
                                  'artist': 'Q5',
                                  'artists': 'Q5',
                                  'album': 'Q482994',
                                  'genres': 'Q188451'}   # Q188451:music genre


BANNED_PHRASES = set(
    stopwords.words('english') + \
    ['music', 'musics', 'name', 'names', 'want', 'wants', 'album', 'albums', 'please', 'who', 'show me', 'tell me', 'find me', 'sing', 'sang',
     'play', 'play me', 'plays', 'track', 'tracks', 'song', 'songs', 'record', 'records', 'recordings', 'album', 'url', 'mount to',
     'something', 'get', 'selections', 'pages', 'isbn', 'isbn numbers', 'average rating', 'count', 'yesterday', 'before today', 'i need to know',
     'resume', 'resumes', 'the', 'search for me', 'search', 'searches', 'yes', 'yeah', 'popular', 'trouble', 'go', 'millisecond', 'good music', 'hear music',
     'h', 'm', 's', 'd', 'y', 'am', 'pm', 'min', 'sec', 'hour', 'year', 'month', 'day', 'us', 'we', 'who', 'what', 'where', 'the',
     'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
     'release', 'released', 'dance', 'dancing', 'need', 'i need', 'i would', ' i will', 'find', 'the list', 'get some', 'af', '1st', '2nd', '3rd',
     'tongue', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'spotify', 'app', 'almond', 'genre',
     'play dj', 'stone', 'sound tracks', 'hi', 'hey', 'tweet', 'all music', 'hello', 'preference', 'top tracks', 'all the good', 'music i', 'id',
     'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'from yesterday', 'show tunes', 'tag', 'ms', 'all good',
     'greatest hits', 'good hits', 'content rating', 'how long', 'actor', 'pg', 'ratings', 'rating', 'rated pg', 'key', 'the nice',
     'keyword', 'keywords', 'subtitle', 'subtitles', 'i want that', 'shorter', 'duration', 'num', 'hope', 'rm', 'michelin', 'michelin star', 'michelin stars',
     'reservations', 'zip code', 'zipcode', 'smoke', 'smoking', 'luxury', 'bar', 'bars', 'kitchen', 'cafe', 'cafes', 'coffee', 'where i am',
     'email']
)

BANNED_REGEXES = [re.compile(r'\d (star|rating)'), re.compile(r'\dth'), re.compile(r'a \d'),
                re.compile(r'\d (hour|min|sec|minute|second|day|month|year)s?'), re.compile(r'this (hour|min|sec|minute|second|day|month|year)s?')]

def is_banned(word):
    return word in BANNED_PHRASES or any([regex.match(word) for regex in BANNED_REGEXES])

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
