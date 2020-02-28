import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


def _default_unk_index():
    return 0


class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=('<pad>',)):
        """Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        counter.update(specials)

        self.stoi = defaultdict(_default_unk_index)
        self.stoi.update({tok: i for i, tok in enumerate(specials)})
        self.itos = list(specials)

        counter.subtract({tok: counter[tok] for tok in specials})
        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
            self.stoi[word] = len(self.itos) - 1

    def __eq__(self, other):
        if self.freqs != other.freqs or self.stoi != other.stoi or self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

    @staticmethod
    def build_from_data(field_names, *args, unk_token=None, pad_token=None, init_token=None, eos_token=None, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            sources += [getattr(ex, name) for name in field_names for ex in arg]
        for data in sources:
            counter.update(data)
        specials = [unk_token, pad_token, init_token, eos_token]
        specials = [tok for tok in specials if tok is not None]
        return Vocab(counter, specials=specials, **kwargs)
