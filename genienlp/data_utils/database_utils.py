import unicodedata
import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


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
