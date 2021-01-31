import unicodedata
import re

import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


DOMAIN_TYPE_MAPPING = dict()

## SQA


DOMAIN_TYPE_MAPPING['music'] = {'MusicRecording': 'Q7366',
                                'Person': 'Q5',
                                'MusicAlbum': 'Q482994',
                                'inLanguage': 'Q315'}   # Q5:human, Q7366:song, Q482994:album


# TODO actor and director should be handled differently
DOMAIN_TYPE_MAPPING['movies'] = {'Movie': 'Q11424',
                                 'Person.creator': 'Q2500638',
                                 'Person.director': 'Q3455803',
                                 'Person.actor': 'Q33999'}   # Q11424:film

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
     'greatest hits', 'good hits']
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
