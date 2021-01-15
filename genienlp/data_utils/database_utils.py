import unicodedata
import re

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


DOMAIN_TYPE_MAPPING = dict()

## SQA
DOMAIN_TYPE_MAPPING['music'] = {'MusicRecording': 'Q7366', 'Person': 'Q5', 'MusicAlbum': 'Q208569', 'inLanguage': 'Q315'}   # Q5:human, Q7366:song, Q208569:studio_album

# TODO actor and director should be handled differently
DOMAIN_TYPE_MAPPING['movies'] = {'Movie': 'Q11424', 'Person.creator': 'Q2500638', 'Person.director': 'Q3455803', 'Person.actor': 'Q33999'}   # Q11424:film

DOMAIN_TYPE_MAPPING['books'] = {'Book': 'Q571', 'Person': 'Q5', 'inLanguage': 'Q315', 'award': 'Q618779'}  # Q571:book, Q315:language, Q618779:award

DOMAIN_TYPE_MAPPING['linkedin'] = {'Organization': 'Q43229','Person': 'Q5', 'addressLocality': 'Q2221906', 'award': 'Q618779'} # Q2221906:geographic_location

DOMAIN_TYPE_MAPPING['restaurants'] = {'Restaurant': 'Q571', 'Person': 'Q5', 'servesCuisine': 'Q1778821', 'Location': 'Q2221906',
                                      'postalCode': 'Q37447', 'ratingValue': 'Q2283373', 'reviewCount': 'Q265158'}   # Q2283373:restaurant_rating, Q265158:review, Q1778821:cuisine

DOMAIN_TYPE_MAPPING['hotels'] = {'Hotel': 'Q571', 'LocationFeatureSpecification': 'Q5912147', 'Location': 'Q2221906',
                                 'CheckinTime': 'Q1068755', 'CheckoutTime': 'Q56353377', 'ratingValue': 'Q2976556'}   # Q5912147:hotel_amenity, Q2976556:hotel_rating


## Dialogues
DOMAIN_TYPE_MAPPING['spotify'] = {'id': 'Q134556', 'song': 'Q7366', 'artist': 'Q5',
                                  'artists': 'Q5', 'album': 'Q208569', 'genres': 'Q188451'}   # Q188451:music genre

# Order of types should not be changed (new types can be appended)
TYPES = ('song_name', 'song_artist', 'song_album', 'song_genre')


BANNED_WORDS = set(
    stopwords.words('english') + \
    ['music', 'musics', 'name', 'names', 'want', 'wants', 'album', 'albums', 'please', 'who', 'show me', 'tell me', 'find me',
     'play', 'play me', 'plays', 'track', 'tracks', 'song', 'songs', 'record', 'records', 'recordings', 'album',
     'something', 'get', 'selections',
     'resume', 'resumes', 'the', 'search for me', 'search', 'searches', 'yes', 'yeah', 'popular',
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
