import re

quoted_pattern_maybe_space = re.compile(r'\"\s?([^"]*?)\s?\"')
device_pattern = re.compile(r'\s@([\w\.]+)\s')

ISO_to_LANG = {'en': 'English', 'en-US': 'English', 'fa': 'Persian', 'it': 'Italian', 'zh': 'Chinese',
               'hr': 'Croatian', 'ja': 'Japanese', 'ko': 'Korean', 'ru': 'Russian', 'es': 'Spanish',
               'sv': 'Swedish', 'tr': 'Turkish', 'hi': 'Hindi', 'fr': 'French', 'de': 'German',
               'pl': 'Polsih', 'ar': 'Arabic', 'vi': 'Vietnamese', 'ji': 'Yiddish', 'pt': 'Portuguese',
               'el': 'Greek', 'he': 'Hebrew', 'si': 'Sinhala', 'ta': 'Tamil', 'fi': 'Finnish', 'cs': 'Czech',
               'no': 'Norwegian', 'tl': 'Filipino', 'da': 'Danish'}


CJK_RANGES = [
    (ord(u"\u3300"), ord(u"\u33ff")), (ord(u"\ufe30"), ord(u"\ufe4f")),   # compatibility ideographs
    (ord(u"\uf900"), ord(u"\ufaff")), (ord(u"\U0002F800"), ord(u"\U0002fa1f")),   # compatibility ideographs
    (ord(u'\u3040'), ord(u'\u309f')),   # Japanese Hiragana
    (ord(u"\u30a0"), ord(u"\u30ff")),   # Japanese Katakana
    (ord(u"\u2e80"), ord(u"\u2eff")),   # cjk radicals supplement
    (ord(u"\u4e00"), ord(u"\u9fff")),
    (ord(u"\u3400"), ord(u"\u4dbf")),
    (ord(u"\U00020000"), ord(u"\U0002a6df")),
    (ord(u"\U0002a700"), ord(u"\U0002b73f")),
    (ord(u"\U0002b740"), ord(u"\U0002b81f")),
    (ord(u"\U0002b820"), ord(u"\U0002ceaf"))
]

CJK_ADDONS = [ord(u"\u3001")]


def is_cjk_char(cp):
  return cp in CJK_ADDONS or any([range[0] <= cp <= range[1] for range in CJK_RANGES])

def is_entity(token):
    return token[0].isupper()

def is_device(token):
    return token[0] == '@'

def process_id(ex):
    # Example instance
    if isinstance(ex.example_id, str):
        id_ = ex.example_id.rsplit('/', 1)
    # NumericalizedExample instance
    else:
        assert isinstance(ex.example_id, list)
        assert len(ex.example_id) == 1
        id_ = ex.example_id[0].rsplit('/', 1)
    id_ = id_[0] if len(id_) == 1 else id_[1]
    # translated
    if id_[0] == 'T':
        id_ = id_[1:]
    return id_



