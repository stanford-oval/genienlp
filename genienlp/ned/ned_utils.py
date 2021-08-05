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


import re
import unicodedata

import nltk
from nltk.corpus import stopwords

from genienlp import ned

nltk.download('stopwords', quiet=True)

BANNED_PHRASES = set(
    stopwords.words('english')
    + [
        'music',
        'musics',
        'name',
        'names',
        'want',
        'wants',
        'album',
        'albums',
        'please',
        'who',
        'show me',
        'tell me',
        'find me',
        'sing',
        'sang',
        'play',
        'play me',
        'plays',
        'track',
        'tracks',
        'song',
        'songs',
        'record',
        'records',
        'recordings',
        'album',
        'url',
        'mount to',
        'something',
        'get',
        'selections',
        'pages',
        'isbn',
        'isbn numbers',
        'average rating',
        'count',
        'yesterday',
        'before today',
        'i need to know',
        'resume',
        'resumes',
        'the',
        'search for me',
        'search',
        'searches',
        'yes',
        'yeah',
        'popular',
        'trouble',
        'go',
        'millisecond',
        'good music',
        'hear music',
        'h',
        'm',
        's',
        'd',
        'y',
        'am',
        'pm',
        'min',
        'sec',
        'hour',
        'year',
        'month',
        'day',
        'us',
        'we',
        'who',
        'what',
        'where',
        'the',
        'mon',
        'tue',
        'wed',
        'thu',
        'fri',
        'sat',
        'sun',
        'monday',
        'tuesday',
        'wednesday',
        'thursday',
        'friday',
        'saturday',
        'sunday',
        'release',
        'released',
        'dance',
        'dancing',
        'need',
        'i need',
        'i would',
        ' i will',
        'find',
        'the list',
        'get some',
        'af',
        '1st',
        '2nd',
        '3rd',
        'tongue',
        'zero',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine',
        'spotify',
        'app',
        'almond',
        'genre',
        'play dj',
        'stone',
        'sound tracks',
        'hi',
        'hey',
        'tweet',
        'all music',
        'hello',
        'preference',
        'top tracks',
        'all the good',
        'music i',
        'id',
        'jan',
        'feb',
        'mar',
        'apr',
        'may',
        'jun',
        'jul',
        'aug',
        'sep',
        'oct',
        'nov',
        'dec',
        'from yesterday',
        'show tunes',
        'tag',
        'ms',
        'all good',
        'greatest hits',
        'good hits',
        'content rating',
        'how long',
        'actor',
        'pg',
        'ratings',
        'rating',
        'rated pg',
        'key',
        'the nice',
        'keyword',
        'keywords',
        'subtitle',
        'subtitles',
        'i want that',
        'shorter',
        'duration',
        'num',
        'hope',
        'rm',
        'michelin',
        'michelin star',
        'michelin stars',
        'reservations',
        'zip code',
        'zipcode',
        'smoke',
        'smoking',
        'luxury',
        'bar',
        'bars',
        'kitchen',
        'cafe',
        'cafes',
        'coffee',
        'where i am',
        'email',
        'motion sensor',
        'temperature sensor',
        'publication date',
        'home address',
        'home assistant',
    ]
)

BANNED_REGEXES = [
    re.compile(r'\d (star|rating)'),
    re.compile(r'\dth'),
    re.compile(r'a \d'),
    re.compile(r'\d (hour|min|sec|minute|second|day|month|year)s?'),
    re.compile(r'this (hour|min|sec|minute|second|day|month|year)s?'),
]


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


def reverse_bisect_left(a, x, lo=None, hi=None):
    """
    Locate the insertion point for x in a to maintain its reverse sorted order
    """
    if lo is None:
        lo = 0
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x > a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def init_ned_model(args, ned_class_name=None):
    ned_model = None
    if args.do_ned:
        if ned_class_name is None:
            if args.ned_retrieve_method == 'bootleg':
                ned_class_name = 'Bootleg'
            elif args.ned_retrieve_method == 'bootleg-annotator':
                ned_class_name = 'BootlegAnnotator'
            elif args.ned_retrieve_method == 'naive':
                ned_class_name = 'NaiveEntityLinker'
            elif args.ned_retrieve_method == 'type-oracle':
                ned_class_name = 'TypeOracleEntityLinker'
            elif args.ned_retrieve_method == 'entity-oracle':
                ned_class_name = 'EntityOracleEntityLinker'
            else:
                raise ValueError('Invalid ned_retrieve_method')
        ned_class = getattr(ned, ned_class_name)
        ned_model = ned_class(args)
    return ned_model
