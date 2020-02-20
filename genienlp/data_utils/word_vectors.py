import array
import io
import logging
import os
import zipfile
import numpy as np
import gzip
import shutil

import six
from six.moves.urllib.request import urlretrieve
import torch
from tqdm import tqdm
import tarfile

from .hash_table import HashTable

logger = logging.getLogger(__name__)
MAX_WORD_LENGTH = 100


pretrained_aliases = {
    "charngram.100d": lambda: CharNGram(),
    "fasttext.en.300d": lambda: FastText(language="en"),
    "fasttext.simple.300d": lambda: FastText(language="simple"),
    "glove.42B.300d": lambda: GloVe(name="42B", dim="300"),
    "glove.840B.300d": lambda: GloVe(name="840B", dim="300"),
    "glove.twitter.27B.25d": lambda: GloVe(name="twitter.27B", dim="25"),
    "glove.twitter.27B.50d": lambda: GloVe(name="twitter.27B", dim="50"),
    "glove.twitter.27B.100d": lambda: GloVe(name="twitter.27B", dim="100"),
    "glove.twitter.27B.200d": lambda: GloVe(name="twitter.27B", dim="200"),
    "glove.6B.50d": lambda: GloVe(name="6B", dim="50"),
    "glove.6B.100d": lambda: GloVe(name="6B", dim="100"),
    "glove.6B.200d": lambda: GloVe(name="6B", dim="200"),
    "glove.6B.300d": lambda: GloVe(name="6B", dim="300")
}


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


class Vectors(object):

    def __init__(self, name, cache='.vector_cache',
                 url=None, unk_init=torch.Tensor.zero_):
        """Arguments:
               name: name of the file that contains the vectors
               cache: directory for cached vectors
               url: url for download if vectors not found in cache
               unk_init (callback): by default, initalize out-of-vocabulary word vectors
                   to zero vectors; can be any function that takes in a Tensor and
                   returns a Tensor of the same size
         """
        self.unk_init = unk_init
        self.cache(name, cache, url=url)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def cache(self, name, cache, url=None):
        if os.path.isfile(name):
            path = name
            path_vectors_np = os.path.join(cache, os.path.basename(name)) + '.vectors.npy'
            path_itos_np = os.path.join(cache, os.path.basename(name)) + '.itos.npy'
            path_table_np = os.path.join(cache, os.path.basename(name)) + '.table.npy'
        else:
            path = os.path.join(cache, name)
            path_vectors_np = path + '.vectors.npy'
            path_itos_np = path + '.itos.npy'
            path_table_np = path + '.table.npy'

        if not os.path.isfile(path_vectors_np):
            if not os.path.isfile(path) and url:
                logger.info('Downloading vectors from {}'.format(url))
                if not os.path.exists(cache):
                    os.makedirs(cache)
                dest = os.path.join(cache, os.path.basename(url))
                if not os.path.isfile(dest):
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                        urlretrieve(url, dest, reporthook=reporthook(t))
                logger.info('Extracting vectors into {}'.format(cache))
                ext = os.path.splitext(dest)[1][1:]
                if ext == 'zip':
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(cache)
                elif dest.endswith('.tar.gz'):
                    with tarfile.open(dest, 'r:gz') as tar:
                        tar.extractall(path=cache)
                elif ext == 'gz':
                    with gzip.open(dest, 'rb') as fin, open(path, 'wb') as fout:
                        shutil.copyfileobj(fin, fout)
            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            # str call is necessary for Python 2/3 compatibility, since
            # argument must be Python 2 str (Python 3 bytes) or
            # Python 3 str (Python 2 unicode)
            itos, vectors, dim = [], array.array(str('d')), None

            # Try to read the whole file with utf-8 encoding.
            binary_lines = False
            try:
                with io.open(path, encoding="utf8") as f:
                    lines = [line for line in f]
            # If there are malformed lines, read in binary mode
            # and manually decode each word from utf-8
            except:
                logger.warning("Could not read {} as UTF8 file, "
                               "reading file as bytes and skipping "
                               "words with malformed UTF8.".format(path))
                with open(path, 'rb') as f:
                    lines = [line for line in f]
                binary_lines = True

            logger.info("Loading vectors from {}".format(path))
            vectors = None
            i = 0
            for line in tqdm(lines, total=len(lines)):
                # Explicitly splitting on " " is important, so we don't
                # get rid of Unicode non-breaking spaces in the vectors.
                entries = line.rstrip().split(b" " if binary_lines else " ")

                word, entries = entries[0], entries[1:]
                if dim is None and len(entries) > 1:
                    dim = len(entries)
                    vectors = np.zeros((len(lines), dim), dtype=np.float32)
                elif len(entries) == 1:
                    logger.warning("Skipping token {} with 1-dimensional "
                                   "vector {}; likely a header".format(word, entries))
                    continue
                elif dim != len(entries):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, but previously "
                        "read vectors have {} dimensions. All vectors must have "
                        "the same number of dimensions.".format(word, len(entries), dim))

                if binary_lines:
                    try:
                        if isinstance(word, six.binary_type):
                            word = word.decode('utf-8')
                    except:
                        logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                        continue

                if len(word) > MAX_WORD_LENGTH:
                    continue
                vectors[i] = [float(x) for x in entries]
                i += 1
                itos.append(word)
            del lines

            # we dropped some words because they were too long, so now vectors
            # has some empty entries at the end
            assert len(itos) <= vectors.shape[0]
            vectors = vectors[:len(itos)]

            self.stoi = HashTable(itos)
            self.itos = self.stoi.itos
            del itos
            assert self.itos.shape[0] == vectors.shape[0]

            print('Saving vectors to {}'.format(path_vectors_np))

            np.save(path_vectors_np, vectors)
            np.save(path_itos_np, self.itos)
            np.save(path_table_np, self.stoi.table)

            self.vectors = torch.from_numpy(vectors)
            self.dim = dim
        else:
            logger.info('Loading vectors from {}'.format(path_vectors_np))

            vectors = np.load(path_vectors_np, mmap_mode='r')
            itos = np.load(path_itos_np, mmap_mode='r')
            table = np.load(path_table_np, mmap_mode='r')
            self.stoi = HashTable(itos, table)
            self.itos = self.stoi.itos
            self.vectors = torch.from_numpy(vectors)
            self.dim = self.vectors.size()[1]


class GloVe(Vectors):
    url = {
        '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }

    def __init__(self, name='840B', dim=300, **kwargs):
        url = self.url[name]
        name = 'glove.{}.{}d.txt'.format(name, str(dim))
        super(GloVe, self).__init__(name, url=url, **kwargs)


class FastText(Vectors):
    url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{}.300.vec.gz'

    def __init__(self, language="en", **kwargs):
        url = self.url_base.format(language)
        name = os.path.basename(url)[:-3]
        super(FastText, self).__init__(name, url=url, **kwargs)


class CharNGram(Vectors):
    name = 'charNgram.txt'
    url = ('http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/'
           'jmt_pre-trained_embeddings.tar.gz')

    def __init__(self, **kwargs):
        super(CharNGram, self).__init__(self.name, url=self.url, **kwargs)

    def __getitem__(self, token):
        vector = torch.Tensor(1, self.dim).zero_()
        if token == "<unk>":
            return self.unk_init(vector)
        # These literals need to be coerced to unicode for Python 2 compatibility
        # when we try to join them with read ngrams from the files.
        chars = ['#BEGIN#'] + list(token) + ['#END#']
        num_vectors = 0
        for n in [2, 3, 4]:
            end = len(chars) - n + 1
            grams = [chars[i:(i + n)] for i in range(end)]
            for gram in grams:
                gram_key = '{}gram-{}'.format(n, ''.join(gram))
                if gram_key in self.stoi:
                    vector += self.vectors[self.stoi[gram_key]]
                    num_vectors += 1
        if num_vectors > 0:
            vector /= num_vectors
        else:
            vector = self.unk_init(vector)
        return vector
