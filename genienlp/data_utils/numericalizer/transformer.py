#
# Copyright (c) 2019-2020 The Board of Trustees of the Leland Stanford Junior University
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
import collections
import os
import torch

from .decoder_vocab import DecoderVocabulary
from .masked_tokenizer import MaskedBertTokenizer, MaskedXLMRobertaTokenizer
from .sequential_field import SequentialField
from transformers.tokenization_xlnet import SPIECE_UNDERLINE


class TransformerNumericalizer(object):
    """
    Numericalizer that uses Tokenizers from huggingface's transformers library.
    """

    def __init__(self, pretrained_tokenizer, config, max_generative_vocab, cache=None, fix_length=None):
        self.config = config
        self._pretrained_name = pretrained_tokenizer
        self.max_generative_vocab = max_generative_vocab
        self._cache = cache
        self._tokenizer = None
        self.fix_length = fix_length

    @property
    def vocab(self):
        return self._tokenizer

    @property
    def num_tokens(self):
        return len(self._tokenizer)

    def load(self, save_dir):
        raise NotImplementedError()

    def save(self, save_dir):
        self._tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, 'decoder-vocab.txt'), 'w') as fp:
            for word in self._decoder_words:
                fp.write(word + '\n')

    def build_vocab(self, vocab_fields, vocab_sets):
        raise NotImplementedError()

    def grow_vocab(self, examples):
        # do a pass over all the data in the dataset and tokenize everything
        # this will add any new tokens that are not to be converted into word-pieces
        for example in examples:
            self._tokenizer.tokenize(example.context, example.context_word_mask)
            self._tokenizer.tokenize(example.question, example.question_word_mask)

        # return no new words - BertEmbedding will resize the embedding regardless
        return []

    def get_special_token_mask(self, token_ids):
        special_tokens_tuple = (self.init_id, self.eos_id, self.pad_id, self.mask_id)
        return list(map(lambda x: 1 if x in special_tokens_tuple else 0, token_ids))

    def _init(self):
        self.pad_first = self._tokenizer.padding_side == 'left'

        self.init_token = self._tokenizer.bos_token
        self.eos_token = self._tokenizer.eos_token
        self.unk_token = self._tokenizer.unk_token
        self.pad_token = self._tokenizer.pad_token
        self.mask_token = self._tokenizer.mask_token

        self.init_id = self._tokenizer.bos_token_id
        self.eos_id = self._tokenizer.eos_token_id
        self.unk_id = self._tokenizer.unk_token_id
        self.pad_id = self._tokenizer.pad_token_id
        self.mask_id = self._tokenizer.mask_token_id
        self.generative_vocab_size = len(self._decoder_words)

        assert self.init_id < self.generative_vocab_size
        assert self.eos_id < self.generative_vocab_size
        assert self.unk_id < self.generative_vocab_size
        assert self.pad_id < self.generative_vocab_size
        assert self.mask_id < self.generative_vocab_size

        self.decoder_vocab = DecoderVocabulary(self._decoder_words, self._tokenizer,
                                               pad_token=self.pad_token, eos_token=self.eos_token)

    def encode(self, minibatch, decoder_vocab, device=None):
        assert isinstance(minibatch, list)

        # apply word-piece tokenization to everything first
        wp_tokenized = []
        for tokens, mask in minibatch:
            wp_tokenized.append(self._tokenizer.tokenize(tokens, mask))

        if self.fix_length is None:
            max_len = max(len(x) for x in wp_tokenized)
        else:
            max_len = self.fix_length
        padded = []
        lengths = []
        numerical = []
        decoder_numerical = []
        for wp_tokens in wp_tokenized:
            if self.pad_first:
                padded_example = [self.pad_token] * max(0, max_len - len(wp_tokens)) + \
                                 [self.init_token] + \
                                 list(wp_tokens[:max_len]) + \
                                 [self.eos_token]
            else:
                padded_example = [self.init_token] + \
                                 list(wp_tokens[:max_len]) + \
                                 [self.eos_token] + \
                                 [self.pad_token] * max(0, max_len - len(wp_tokens))

            padded.append(padded_example)
            lengths.append(len(wp_tokens) + 2)

            numerical.append(self._tokenizer.convert_tokens_to_ids(padded_example))
            decoder_numerical.append([decoder_vocab.encode(word) for word in padded_example])

        length = torch.tensor(lengths, dtype=torch.int32, device=device)
        numerical = torch.tensor(numerical, dtype=torch.int64, device=device)
        decoder_numerical = torch.tensor(decoder_numerical, dtype=torch.int64, device=device)

        return SequentialField(length=length, value=numerical, limited=decoder_numerical)

    def decode(self, tensor):
        return self._tokenizer.convert_ids_to_tokens(tensor)

    def reverse(self, batch, detokenize, field_name=None):
        raise NotImplementedError()


class XLMRobertaNumericalizer(TransformerNumericalizer):

    def __init__(self, pretrained_tokenizer, config, max_generative_vocab, cache=None, fix_length=None):
        super().__init__(pretrained_tokenizer, config, max_generative_vocab, cache, fix_length)


    def load(self, save_dir):
        self._tokenizer = MaskedXLMRobertaTokenizer.from_pretrained(save_dir, config=self.config, cache_dir=self._cache)
        # HACK we cannot save the tokenizer without this
        del self._tokenizer.init_kwargs['config']

        with open(os.path.join(save_dir, 'decoder-vocab.txt'), 'r') as fp:
            self._decoder_words = [line.rstrip('\n') for line in fp]

        self._init()


    def build_vocab(self, vocab_fields, vocab_sets):
        self._tokenizer = MaskedXLMRobertaTokenizer.from_pretrained(self._pretrained_name, config=self.config,
                                                              cache_dir=self._cache)
        # HACK we cannot save the tokenizer without this
        del self._tokenizer.init_kwargs['config']

        # ensure that init, eos, unk and pad are set
        # this method has no effect if the tokens are already set according to the tokenizer class
        self._tokenizer.add_special_tokens({
            'bos_token': "<s>",
            'eos_token': "</s>",
            'unk_token': "<unk>",
            'pad_token': "<pad>",
            'mask_token': "<mask>",
            'cls_token': "<s>",
        })

        # do a pass over all the data in the dataset
        # in this pass, we
        # 1) tokenize everything, to ensure we account for all added tokens
        # 2) we construct a counter of wordpieces in the answers, for the decoder vocabulary
        decoder_words = collections.Counter()
        for dataset in vocab_sets:
            for example in dataset:
                decoder_words.update(self._tokenizer.tokenize(example.context, example.context_word_mask))
                decoder_words.update(self._tokenizer.tokenize(example.question, example.question_word_mask))
                decoder_words.update(self._tokenizer.tokenize(example.answer, example.answer_word_mask))

        self._decoder_words = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"] + \
                              [word for word, _freq in decoder_words.most_common(self.max_generative_vocab)]

        self._init()


    def reverse(self, batch, detokenize, field_name=None):
        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        def is_entity(token):
            return token[0].isupper()

        def reverse_one(tensor, field_name):
            tokens = []

            in_string = False
            # trim up to EOS, remove other special stuff, and undo wordpiece tokenization
            for token in self.decode(tensor):
                if token == self.eos_token:
                    break
                if token in (self.init_token, self.pad_token):
                    continue
                if token.startswith(SPIECE_UNDERLINE):
                        tokens.append(token[1:])
                elif len(tokens) == 0:
                    tokens.append(token)
                else:
                    if field_name == 'answer':
                        if token == '"':
                            in_string = not in_string
                            tokens.append(token)
                            continue
                        if in_string:
                            tokens[-1] += token
                        else:
                            tokens.append(token)

                    else:
                        if is_entity(token):
                            tokens.append(token)
                        else:
                            tokens[-1] += token

            return detokenize(tokens, field_name=field_name)

        return [reverse_one(tensor, field_name) for tensor in batch]


class BertNumericalizer(TransformerNumericalizer):
    """
    Numericalizer that uses BertTokenizer from huggingface's transformers library.
    """

    def __init__(self, pretrained_tokenizer, config, max_generative_vocab, cache=None, fix_length=None):
        super().__init__(pretrained_tokenizer, config, max_generative_vocab, cache, fix_length)


    def load(self, save_dir):
        self._tokenizer = MaskedBertTokenizer.from_pretrained(save_dir, config=self.config, cache_dir=self._cache)
        # HACK we cannot save the tokenizer without this
        del self._tokenizer.init_kwargs['config']

        with open(os.path.join(save_dir, 'decoder-vocab.txt'), 'r') as fp:
            self._decoder_words = [line.rstrip('\n') for line in fp]

        self._init()


    def build_vocab(self, vocab_fields, vocab_sets):
        self._tokenizer = MaskedBertTokenizer.from_pretrained(self._pretrained_name, config=self.config,
                                                              cache_dir=self._cache)
        # HACK we cannot save the tokenizer without this
        del self._tokenizer.init_kwargs['config']

        # ensure that init, eos, unk and pad are set
        # this method has no effect if the tokens are already set according to the tokenizer class
        self._tokenizer.add_special_tokens({
            'bos_token': '[CLS]',
            'eos_token': '[SEP]',
            'unk_token': '[UNK]',
            'pad_token': '[PAD]',
            'mask_token': '[MASK]'
        })

        # do a pass over all the data in the dataset
        # in this pass, we
        # 1) tokenize everything, to ensure we account for all added tokens
        # 2) we construct a counter of wordpieces in the answers, for the decoder vocabulary
        decoder_words = collections.Counter()
        for dataset in vocab_sets:
            for example in dataset:
                decoder_words.update(self._tokenizer.tokenize(example.context, example.context_word_mask))
                decoder_words.update(self._tokenizer.tokenize(example.question, example.question_word_mask))
                decoder_words.update(self._tokenizer.tokenize(example.answer, example.answer_word_mask))

        self._decoder_words = ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]'] + \
                              [word for word, _freq in decoder_words.most_common(self.max_generative_vocab)]

        self._init()


    def reverse(self, batch, detokenize, field_name=None):
        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        def reverse_one(tensor):
            tokens = []

            # trim up to EOS, remove other special stuff, and undo wordpiece tokenization
            for token in self.decode(tensor):
                if token == self.eos_token:
                    break
                if token in (self.init_token, self.pad_token):
                    continue
                if token.startswith('##'):
                    if len(tokens) == 0:
                        tokens.append(token[2:])
                    else:
                        tokens[-1] += token[2:]
                else:
                    tokens.append(token)

            return detokenize(tokens, field_name=field_name)

        return [reverse_one(tensor) for tensor in batch]
