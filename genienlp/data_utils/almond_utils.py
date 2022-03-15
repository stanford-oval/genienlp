#
# Copyright (c) 2020, The Board of Trustees of the Leland Stanford Junior University
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

from .progbar import progress_bar

quoted_pattern_maybe_space = re.compile(r'\"\s?([^"]*?)\s?\"')
quoted_pattern_with_space = re.compile(r'\"\s([^"]*?)\s\"')
device_pattern = re.compile(r'\s@([\w\.]+)\s')
entity_regex = re.compile('<e>.*?</e>')
token_type_regex = re.compile('(.*?) \( (.*?) \)')

ISO_to_LANG = {
    'en': 'English',
    'en-US': 'English',
    'fa': 'Persian',
    'it': 'Italian',
    'zh': 'Chinese',
    'hr': 'Croatian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ru': 'Russian',
    'es': 'Spanish',
    'sv': 'Swedish',
    'tr': 'Turkish',
    'hi': 'Hindi',
    'fr': 'French',
    'de': 'German',
    'pl': 'Polsih',
    'ar': 'Arabic',
    'vi': 'Vietnamese',
    'ji': 'Yiddish',
    'pt': 'Portuguese',
    'el': 'Greek',
    'he': 'Hebrew',
    'si': 'Sinhala',
    'ta': 'Tamil',
    'fi': 'Finnish',
    'cs': 'Czech',
    'no': 'Norwegian',
    'tl': 'Filipino',
    'da': 'Danish',
}

NUMBER_MAPPING = {
    'en': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
    'fa': ('۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'),
}

ENGLISH_MONTH_MAPPING = {
    '1': ('Jan', 'January'),
    '2': ('Feb', 'February'),
    '3': ('Mar', 'March'),
    '4': ('Apr', 'April'),
    '5': ('May',),
    '6': ('Jun', 'June'),
    '7': ('Jul', 'July'),
    '8': ('Aug', 'August'),
    '9': ('Sep', 'September'),
    '10': ('Oct', 'October'),
    '11': ('Nov', 'November'),
    '12': ('Dec', 'December'),
}


CJK_RANGES = [
    (ord(u"\u3300"), ord(u"\u33ff")),
    (ord(u"\ufe30"), ord(u"\ufe4f")),  # compatibility ideographs
    (ord(u"\uf900"), ord(u"\ufaff")),
    (ord(u"\U0002F800"), ord(u"\U0002fa1f")),  # compatibility ideographs
    (ord(u'\u3040'), ord(u'\u309f')),  # Japanese Hiragana
    (ord(u"\u30a0"), ord(u"\u30ff")),  # Japanese Katakana
    (ord(u"\u2e80"), ord(u"\u2eff")),  # cjk radicals supplement
    (ord(u"\u4e00"), ord(u"\u9fff")),
    (ord(u"\u3400"), ord(u"\u4dbf")),
    (ord(u"\U00020000"), ord(u"\U0002a6df")),
    (ord(u"\U0002a700"), ord(u"\U0002b73f")),
    (ord(u"\U0002b740"), ord(u"\U0002b81f")),
    (ord(u"\U0002b820"), ord(u"\U0002ceaf")),
]

CJK_ADDONS = [ord(u"\u3001")]


def is_cjk_char(cp):
    return cp in CJK_ADDONS or any([range[0] <= cp <= range[1] for range in CJK_RANGES])


ENTITY_REGEX = re.compile('^[A-Z]+_')


def is_entity(token):
    return ENTITY_REGEX.match(token) is not None


def is_device(token):
    return token[0] == '@'


def is_entity_marker(token):
    return token.startswith('^^')


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


def tokenize_cjk_chars(sentence):
    output = []
    i = 0
    while i < len(sentence):
        output.append(sentence[i])
        if is_cjk_char(ord(sentence[i])) and i + 1 < len(sentence) and sentence[i + 1] != ' ':
            output.append(' ')
        elif not is_cjk_char(ord(sentence[i])) and i + 1 < len(sentence) and is_cjk_char(ord(sentence[i + 1])):
            output.append(' ')
        i += 1

    output = "".join(output)
    output = output.replace('  ', ' ')

    return output


def detokenize_cjk_chars(sentence):
    output = []
    i = 0
    while i < len(sentence):
        output.append(sentence[i])
        # skip space after cjk chars only if followed by another cjk char
        if (
            is_cjk_char(ord(sentence[i]))
            and i + 1 < len(sentence)
            and sentence[i + 1] == ' '
            and i + 2 < len(sentence)
            and is_cjk_char(ord(sentence[i + 2]))
        ):
            i += 2
        else:
            i += 1

    return "".join(output)


def chunk_file(input_src, chunk_files, chunk_size, num_chunks):
    chunk_id = 0
    num_lines_in_chunk = 0
    all_out_files = [open(chunk_files[chunk_id], 'w') for chunk_id in range(num_chunks)]
    with open(input_src, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            all_out_files[chunk_id].write(line)
            num_lines_in_chunk += 1
            if num_lines_in_chunk == chunk_size:
                chunk_id += 1
                num_lines_in_chunk = 0
                if chunk_id == num_chunks:
                    break
    for file in all_out_files:
        file.close()


def create_examples_from_file(args):
    path = args['in_file']
    chunk_size = args['chunk_size']
    dir_name = args['dir_name']
    example_batch_size = args['example_batch_size']
    make_process_example = args['make_process_example']
    kwargs = args['kwargs']

    chunk_examples = []

    batch = []
    last_batch = False
    for i, line in progress_bar(enumerate(open(path, 'r', encoding='utf-8')), desc='Reading dataset'):
        parts = line.strip().split('\t')
        batch.append(parts)
        if len(chunk_examples) + example_batch_size > chunk_size:
            # trim batch
            batch = batch[: chunk_size - len(chunk_examples)]
            last_batch = True
        if len(batch) % example_batch_size != 0 and not last_batch:
            continue

        # TODO remote database lookup is faster when multiple examples are sent in one HTTP request
        # For now we only do one at a time; support processing examples as a batch
        assert len(batch) == 1
        batch = batch[0]
        examples = make_process_example(batch, dir_name, **kwargs)
        if isinstance(examples, list):
            # account for extra examples created when using --translate_example_split
            chunk_size += len(examples) - 1
            chunk_examples.extend(examples)
        else:
            chunk_examples.append(examples)
        batch = []
        if len(chunk_examples) >= chunk_size:
            break

    return chunk_examples


def inside_spans(start, spans):
    if not spans:
        return False
    for span in spans:
        if span[0] <= start < span[1]:
            return True
    return False


def return_sentences(text, regex_pattern, src_char_spans, is_cjk=False):
    sentences = []
    cur = 0
    for m in re.finditer(regex_pattern, text, flags=re.U):
        if not inside_spans(m.start(0), src_char_spans):
            sentences.append(text[cur : m.start(0) + (1 if is_cjk else 0)])
            cur = m.end(0)
    if cur != len(text):
        sentences.append(text[cur:])
    return sentences


def split_text_into_sentences(text, lang, src_char_spans):
    if lang in ['en']:
        sentences = return_sentences(text, '(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[\.!?])\s', src_char_spans)
    elif lang in ['zh', 'ja', 'ko']:
        sentences = return_sentences(text, u'([!！?？。])\s?', src_char_spans, is_cjk=True)
    else:
        import nltk

        nltk.download('punkt', quiet=True)
        sentences = nltk.sent_tokenize(text, language=ISO_to_LANG[lang])

    return sentences


def input_heuristics(s: str, thingtalk=None, is_cased=False, keep_special_tokens=False, keep_tokenized=False):
    """
    Changes the input string so that it is closer to what the pre-trained language models have seen during their training.
    Outputs:
        s: the new string
        reverse_map: a list of special tokens. Can be used to recover the original special_tokens in the string
    """
    s = s.strip()
    s = tokenize(s)

    # Put question mark at the end whenever necessary.
    sentences = [sentence.strip() for sentence in re.split('\s+([.?!:])\s*', s) if len(sentence) > 0]
    # logger.info('sentences = %s', sentences)
    for idx in range(len(sentences)):
        if sentences[idx] in ['.', '?', '!', ':']:
            continue
        if idx == len(sentences) - 1 or sentences[idx + 1] not in ['.', '?', '!', ':']:
            # add the missing punctuation
            if is_question(sentences[idx]):
                sentences[idx] = sentences[idx] + '?'
            else:
                sentences[idx] = sentences[idx] + '.'
        else:
            if is_question(sentences[idx]):
                assert sentences[idx + 1] in ['.', '?', '!', ':']
                sentences[idx + 1] = '?'

        if is_cased:
            # capitalize the first word and parameters
            if thingtalk:
                _, parameters = remove_thingtalk_quotes(thingtalk)
                # logger.info('parameters = ', parameters)
                for p in parameters:
                    capitalized_p = ' '.join([t[0].upper() + t[1:] for t in p.split()])
                    sentences[idx] = sentences[idx].replace(p, capitalized_p)
            sentences[idx] = sentences[idx].replace(' i ', ' I ')
            sentences[idx] = sentences[idx][0].upper() + sentences[idx][1:]

    s = ' '.join(sentences)
    if not keep_tokenized:
        s = detokenize(s)

    if not is_cased:
        s = lower_case(s)

    # replace special tokens with natural-looking examples
    reverse_map = []
    if not keep_special_tokens:
        for spm in special_pattern_mapping:
            s, r = spm.forward(s)
            reverse_map.extend(r)

    return s, reverse_map


def output_heuristics(s: str, reverse_map: list):
    for spm, occurance in reverse_map:
        s = spm.backward(s, occurance)

    s = tokenize(s)
    s = lower_case(s)
    return s


class SpecialTokenMap:
    def __init__(self, pattern, forward_func, backward_func=None):
        """
        Inputs:
            pattern: a regex pattern
            forward_func: a function with signature forward_func(str) -> str
            backward_func: a function with signature backward_func(str) -> list[str]
        """
        if isinstance(forward_func, list):
            self.forward_func = lambda x: forward_func[int(x) % len(forward_func)]
        else:
            self.forward_func = forward_func

        if isinstance(backward_func, list):
            self.backward_func = lambda x: backward_func[int(x) % len(backward_func)]
        else:
            self.backward_func = backward_func

        self.pattern = pattern

    def forward(self, s: str):
        reverse_map = []
        matches = re.finditer(self.pattern, s)
        if matches is None:
            return s, reverse_map
        for match in matches:
            occurrence = match.group(0)
            parameter = match.group(1)
            replacement = self.forward_func(parameter)
            s = s.replace(occurrence, replacement)
            reverse_map.append((self, occurrence))
        return s, reverse_map

    def backward(self, s: str, occurrence: str):
        match = re.match(self.pattern, occurrence)
        parameter = match.group(1)
        if self.backward_func is None:
            list_of_strings_to_match = [self.forward_func(parameter)]
        else:
            list_of_strings_to_match = sorted(self.backward_func(parameter), key=lambda x: len(x), reverse=True)
        for string_to_match in list_of_strings_to_match:
            l_ = [' ' + string_to_match + ' ', string_to_match + ' ', ' ' + string_to_match]
            o_ = [' ' + occurrence + ' ', occurrence + ' ', ' ' + occurrence]
            new_s = s
            for i in range(len(l_)):
                new_s = re.sub(l_[i], o_[i], s, flags=re.IGNORECASE)
                if s != new_s:
                    break
            if s != new_s:
                s = new_s
                break

        return s


special_pattern_mapping = [
    SpecialTokenMap('PHONE_NUMBER_([0-9]+)', ['888-8888', '777-8888']),
    SpecialTokenMap('NUMBER_([0-9]+)', ['2', '3'], [['2', 'two'], ['3', 'three']]),
    SpecialTokenMap('PATH_NAME_([0-9]+)', ['my1folder', 'my2folder']),
    SpecialTokenMap(
        'TIME_([0-9]+)',
        ['5p.m.', '2p.m.'],
        [
            ['5 pm', '5pm', '5:00 pm', '5:00pm', '5p.m.', '5 p.m.', '5:00 p.m.', '5:00', 'five o\'clock', 'five'],
            ['2 pm', '2pm', '2:00 pm', '2:00pm', '2p.m.', '2 p.m.', '2:00 p.m.', '2:00', 'two o\'clock', 'two'],
        ],
    ),
    SpecialTokenMap('EMAIL_ADDRESS_([0-9]+)', ['e1@example.com', 'e2@example.com']),
    SpecialTokenMap('URL_([0-9]+)', ['my1site.com', 'my2site.com']),
    SpecialTokenMap('DATE_([0-9]+)', ['5-6-2015', '8-3-2016']),
    SpecialTokenMap(
        'CURRENCY_([0-9]+)',
        ['$12', '$13'],
        [
            ['$12', 'twelve dollars', '12 dollars', '$ 12', '$ 12.00', '12.00', '12'],
            ['$13', 'thirteen dollars', '13 dollars', '$ 13', '$ 13.00', '13.00', '13'],
        ],
    ),
    SpecialTokenMap('DURATION_([0-9]+)', ['5 weeks', '6 weeks'], [['5 weeks', 'five weeks'], ['6 weeks', 'six weeks']]),
    SpecialTokenMap('LOCATION_([0-9]+)', ['locatio1n', 'locatio2n'], [['locatio1n', 'locat1n'], ['locatio2n', 'locat2n']]),
    # SpecialTokenMap('QUOTED_STRING_([0-9]+)', ['Chinese', 'Italian'], [['Chinese', 'chinese', 'china'], ['Italian', 'italian']]), # TODO change to be more general than cuisine
    # SpecialTokenMap('GENERIC_ENTITY_uk.ac.cam.multiwoz.Restaurant:Restaurant_([0-9]+)', ["restaurant1", "restaurant2", "restaurant3"]) # TODO the only reason we can get away with this unnatural replacement is that actual backward is not going to be called for this
]


def is_question(sentence: str):
    question_words = [
        'which',
        'what',
        'where',
        'how',
        'who',
        'when',
        'is',
        'are',
        'am',
        'can',
        'could',
        'would',
        'will',
        'have',
        'did',
        'do',
        'does',
        'no is',
        'yes is',
    ]
    for w in question_words:
        if sentence.startswith(w + ' '):
            return True
    return False


def remove_thingtalk_quotes(thingtalk):
    quote_values = []
    while True:
        # print('before: ', thingtalk)
        l1 = thingtalk.find('"')
        if l1 < 0:
            break
        l2 = thingtalk.find('"', l1 + 1)
        if l2 < 0:
            # ThingTalk code is not syntactic
            return thingtalk, None
        quote_values.append(thingtalk[l1 + 1 : l2].strip())
        thingtalk = thingtalk[:l1] + '<temp>' + thingtalk[l2 + 1 :]
        # print('after: ', thingtalk)
    thingtalk = thingtalk.replace('<temp>', '""')
    return thingtalk, quote_values


def requote_program(program):

    program = program.split(' ')
    requoted = []

    in_string = False
    begin_index = 0
    i = 0
    while i < len(program):
        token = program[i]
        if token == '"':
            in_string = not in_string
            if in_string:
                begin_index = i + 1
            else:
                span_type, end_index = find_span_type(program, begin_index, i)
                requoted.append(span_type)
                i = end_index

        elif not in_string:
            entity_match = ENTITY_MATCH_REGEX.match(token)
            if entity_match is not None:
                requoted.append(entity_match[1])
            elif token != 'location:':
                requoted.append(token)

        i += 1

    return ' '.join(requoted)


def detokenize(string: str):
    string, exceptions = mask_special_tokens(string)
    tokens = ["'d", "n't", "'ve", "'m", "'re", "'ll", ".", ",", "?", "!", "'s", ")", ":", "-"]
    for t in tokens:
        string = string.replace(' ' + t, t)
    string = string.replace("( ", "(")
    string = string.replace('gon na', 'gonna')
    string = string.replace('wan na', 'wanna')
    string = unmask_special_tokens(string, exceptions)
    return string


def tokenize(string: str):
    string, exceptions = mask_special_tokens(string)
    tokens = ["'d", "n't", "'ve", "'m", "'re", "'ll", ".", ",", "?", "!", "'s", ")", ":"]
    for t in tokens:
        string = string.replace(t, ' ' + t)
    string = string.replace("(", "( ")
    string = string.replace('gonna', 'gon na')
    string = string.replace('wanna', 'wan na')
    string = unmask_special_tokens(string, exceptions)
    string = re.sub('([A-Za-z:_.]+_[0-9]+)-', r'\1 - ', string)  # add space before and after hyphen, e.g. "NUMBER_0-hour"
    string = re.sub('\s+', ' ', string)  # remove duplicate spaces
    return string.strip()


def lower_case(string):
    string, exceptions = mask_special_tokens(string)
    string = string.lower()
    string = unmask_special_tokens(string, exceptions)
    return string


def find_span(haystack, needle):
    for i in range(len(haystack) - len(needle) + 1):
        found = True
        for j in range(len(needle)):
            if haystack[i + j] != needle[j]:
                found = False
                break
        if found:
            return i
    return None


def mask_special_tokens(string: str):
    exceptions = [match.group(0) for match in re.finditer('[A-Za-z:_.]+_[0-9]+', string)]
    for e in exceptions:
        string = string.replace(e, '<temp>', 1)
    return string, exceptions


def unmask_special_tokens(string: str, exceptions: list):
    for e in exceptions:
        string = string.replace('<temp>', e, 1)
    return string


ENTITY_MATCH_REGEX = re.compile('^([A-Z].*)_[0-9]+$')


def find_span_type(program, begin_index, end_index):

    if begin_index > 1 and program[begin_index - 2] == 'location:':
        span_type = 'LOCATION'
    elif end_index == len(program) - 1 or not program[end_index + 1].startswith('^^'):
        span_type = 'QUOTED_STRING'
    else:
        if program[end_index + 1] == '^^tt:hashtag':
            span_type = 'HASHTAG'
        elif program[end_index + 1] == '^^tt:username':
            span_type = 'USERNAME'
        else:
            span_type = 'GENERIC_ENTITY_' + program[end_index + 1][2:]

        end_index += 1

    return span_type, end_index
