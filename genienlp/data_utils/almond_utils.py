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
    # text = '''the . " ${field} " . of . " ${value} " .'''
    if lang in ['en']:
        sentences = return_sentences(text, '(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[\.!?])\s', src_char_spans)
    elif lang in ['zh', 'ja', 'ko']:
        sentences = return_sentences(text, u'([!！?？。])\s?', src_char_spans, is_cjk=True)
    else:
        import nltk

        nltk.download('punkt', quiet=True)
        sentences = nltk.sent_tokenize(text, language=ISO_to_LANG[lang])

    return sentences
