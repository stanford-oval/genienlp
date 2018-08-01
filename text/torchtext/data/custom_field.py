# coding: utf8
import torch
from .field import Field

def tokenizer(s):
    return s.split()

class SimpleReversibleField(Field):
    def __init__(self, **kwargs):
        # if kwargs.get('tokenize') is list:
        #     self.use_revtok = False
        # else:
        #     self.use_revtok = True
        # if kwargs.get('tokenize') is None:
        #     kwargs['tokenize'] = 'revtok'
        if 'unk_token' not in kwargs:
            kwargs['unk_token'] = ' UNK '
        self.use_revtok = False
        kwargs['tokenize'] = tokenizer
        super(SimpleReversibleField, self).__init__(**kwargs)

    def reverse(self, batch, limited=False):
        if self.use_revtok:
            try:
                import revtok
            except ImportError:
                print("Please install revtok.")
                raise
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

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past first eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        if self.use_revtok:
            return [revtok.detokenize(ex) for ex in batch]
        return [' '.join(ex) for ex in batch]
