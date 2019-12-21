# coding: utf8
from copy import deepcopy
from collections import Counter, OrderedDict
import six
import torch
from tqdm import tqdm

from .dataset import Dataset
from .utils import get_tokenizer
from ..vocab import Vocab, SubwordVocab


class Field(object):
    """Defines a datatype together with instructions for converting to Tensor.

    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.

    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.

    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        tensor_type: The torch.Tensor class that represents a batch of examples
            of this kind of data. Default: torch.LongTensor.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list,
            the field's Vocab, and train (a bool).
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy English tokenizer is
            used. Default: str.split.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor types to the appropriate Python
    # numeric type.
    tensor_types = {
        torch.FloatTensor: float,
        torch.cuda.FloatTensor: float,
        torch.DoubleTensor: float,
        torch.cuda.DoubleTensor: float,
        torch.HalfTensor: float,
        torch.cuda.HalfTensor: float,

        torch.ByteTensor: int,
        torch.cuda.ByteTensor: int,
        torch.CharTensor: int,
        torch.cuda.CharTensor: int,
        torch.ShortTensor: int,
        torch.cuda.ShortTensor: int,
        torch.IntTensor: int,
        torch.cuda.IntTensor: int,
        torch.LongTensor: int,
        torch.cuda.LongTensor: int
    }

    def __init__(
            self, sequential=True, use_vocab=True, init_token=None,
            eos_token=None, fix_length=None, tensor_type=torch.LongTensor,
            tokenize=(lambda s: s.split()), include_lengths=False,
            batch_first=False, pad_token="<pad>", unk_token="<unk>",
            pad_first=False, decap=False, numerical=False):
        self.sequential = sequential
        self.numerical = numerical
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.tensor_type = tensor_type
        self.tokenize = get_tokenizer(tokenize)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None
        self.pad_first = pad_first


    def build_vocab(self, field_names, *args, **kwargs):
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
            for x in data:
                if not self.sequential:
                    x = [x]
                counter.update(x)
        specials = [self.unk_token, self.pad_token, self.init_token, self.eos_token]
        specials = list(OrderedDict.fromkeys(tok for tok in specials if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def append_vocab(self, other_field):
        for w, count in other_field.vocab.stoi.items():
            if w not in self.vocab.stoi:
                self.vocab.stoi[w] = len(self.vocab.itos)
                self.vocab.itos.append(w)


class ReversibleField(Field):

    def __init__(self, **kwargs):
        if kwargs.get('tokenize') is list:
            self.use_revtok = False
        else:
            self.use_revtok = True
        if kwargs.get('tokenize') is None:
            kwargs['tokenize'] = 'revtok'
        if self.use_revtok:
            try:
                import revtok
            except ImportError:
                print("Please install revtok.")
                raise
            self.detokenize = revtok.detokenize
        else:
            self.detokenize = None
        super(ReversibleField, self).__init__(**kwargs)

    def reverse(self, batch, detokenize=None, field_name=None, limited=False):
        
        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        if detokenize is not None:
            return [detokenize(ex, field_name=field_name) for ex in batch]
        elif self.detokenize is not None:
            return [self.detokenize(ex) for ex in batch]
        else:
            return [''.join(ex) for ex in batch]
