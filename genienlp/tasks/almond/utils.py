from tqdm import tqdm


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
    id_ = ex.example_id.rsplit('/', 1)
    id_ = id_[0] if len(id_) == 1 else id_[1]
    # translated
    if id_[0] == 'T':
        id_ = id_[1:]
    return id_


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


def process(args):
    path = args['in_file']
    chunk_size = args['chunk_size']
    dir_name = args['dir_name']
    example_batch_size = args['example_batch_size']
    make_process_example = args['make_process_example']
    kwargs = args['kwargs']
    
    chunk_examples = []
    
    batch = []
    last_batch = False
    for i, line in tqdm(enumerate(open(path, 'r', encoding='utf-8')), total=chunk_size):
        parts = line.strip().split('\t')
        batch.append(parts)
        if len(chunk_examples) + example_batch_size > chunk_size:
            # trim batch
            batch = batch[:chunk_size - len(chunk_examples)]
            last_batch = True
        if len(batch) % example_batch_size != 0 and not last_batch:
            continue
        assert len(batch) == 1
        batch = batch[0]
        chunk_examples.append(make_process_example(batch, dir_name, **kwargs))
        batch = []
        if len(chunk_examples) >= chunk_size:
            break
    
    return chunk_examples




