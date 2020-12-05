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
from torch.nn.utils.rnn import pad_sequence

from .decoder_vocab import DecoderVocabulary
from .masked_tokenizer import MaskedBertTokenizer, MaskedXLMRobertaTokenizer
from transformers import BartTokenizer, MBartTokenizer, T5Tokenizer
from .sequential_field import SequentialField
from transformers.file_utils import SPIECE_UNDERLINE


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

    def pad(self, batch):
        """
        batch: a List of List of integers
        """
        #TODO account for left padding models
        return pad_sequence(batch, padding_value=self.pad_id, batch_first=True)

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
        self.cls_token = self._tokenizer.cls_token
        self.sep_token = self._tokenizer.sep_token

        self.init_id = self._tokenizer.bos_token_id
        self.eos_id = self._tokenizer.eos_token_id
        self.unk_id = self._tokenizer.unk_token_id
        self.pad_id = self._tokenizer.pad_token_id
        self.mask_id = self._tokenizer.mask_token_id
        self.cls_id = self._tokenizer.cls_token_id
        self.sep_id = self._tokenizer.sep_token_id
        self.generative_vocab_size = len(self._decoder_words)

        self.decoder_vocab = DecoderVocabulary(self._decoder_words, self._tokenizer,
                                               pad_token=self.pad_token, eos_token=self.eos_token)

    def encode_single(self, minibatch, decoder_vocab, max_length=-1):
        assert isinstance(minibatch, list)

        # apply word-piece tokenization to everything first
        wp_tokenized = []
        for tokens, mask in minibatch:
            wp_tokenized.append(self._tokenizer.tokenize(tokens, mask))

        if max_length > -1:
            max_len = max_length
        elif self.fix_length is None:
            max_len = max(len(x) for x in wp_tokenized)
        else:
            max_len = self.fix_length

        examples = []
        lengths = []
        numerical = []
        decoder_numerical = []
        for wp_tokens in wp_tokenized:
            example = [self.init_token] + \
                                list(wp_tokens[:max_len]) + \
                                [self.eos_token]

            examples.append(example)
            lengths.append(len(example))

            numerical.append(self._tokenizer.convert_tokens_to_ids(example))
            decoder_numerical.append([decoder_vocab.encode(word) for word in example])

        return SequentialField(length=lengths, value=numerical, limited=decoder_numerical)

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
            'sep_token': "</s>",
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

    def encode_pair(self, minibatch, decoder_vocab):
        # apply word-piece tokenization to everything first
        wp_tokenized_a = []
        wp_tokenized_b = []
        for (tokens_a, mask_a), (tokens_b, mask_b) in minibatch:
            wp_tokenized_a.append(self._tokenizer.tokenize(tokens_a, mask_a))
            wp_tokenized_b.append(self._tokenizer.tokenize(tokens_b, mask_b))

        if self.fix_length is None:
            max_len = max(len(wp_a) + len(wp_b) for wp_a, wp_b in zip(wp_tokenized_a, wp_tokenized_b))
        else:
            max_len = self.fix_length

        padded = []
        lengths = []
        numerical = []
        decoder_numerical = []
        for (wp_tokens_a, _), (wp_tokens_b, _) in minibatch:
            # XLM-R uses two sep tokens
            example = [self.init_token] + \
                            list(wp_tokens_a[:max_len]) + \
                            [self.sep_token] + [self.sep_token] + \
                            list(wp_tokens_b[:max_len]) + \
                            [self.eos_token]

            padded.append(example)
            lengths.append(len(example))

            numerical.append(self._tokenizer.convert_tokens_to_ids(example))
            decoder_numerical.append([decoder_vocab.encode(word) for word in example])

        return SequentialField(length=lengths, value=numerical, limited=decoder_numerical)

    def reverse(self, batch, detokenize, field_name=None):
        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        def is_entity(token):
            return token[0].isupper()

        def is_device(token):
            return token[0] == '@'

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
                        if is_entity(token) or is_device(token):
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
            'sep_token': '[SEP]',
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

    def encode_pair(self, minibatch, decoder_vocab):
        # apply word-piece tokenization to everything first
        wp_tokenized_a = []
        wp_tokenized_b = []
        for (tokens_a, mask_a), (tokens_b, mask_b) in minibatch:
            wp_tokenized_a.append(self._tokenizer.tokenize(tokens_a, mask_a))
            wp_tokenized_b.append(self._tokenizer.tokenize(tokens_b, mask_b))

        if self.fix_length is None:
            max_len = max(len(wp_a) + len(wp_b) for wp_a, wp_b in zip(wp_tokenized_a, wp_tokenized_b))
        else:
            max_len = self.fix_length

        padded = []
        lengths = []
        numerical = []
        decoder_numerical = []
        for (wp_tokens_a, _), (wp_tokens_b, _) in minibatch:
            example = [self.init_token] + \
                            list(wp_tokens_a[:max_len]) + \
                            [self.sep_token] + \
                            list(wp_tokens_b[:max_len]) + \
                            [self.eos_token]

            padded.append(example)
            lengths.append(len(example))

            numerical.append(self._tokenizer.convert_tokens_to_ids(example))
            decoder_numerical.append([decoder_vocab.encode(word) for word in example])

        return SequentialField(length=lengths, value=numerical, limited=decoder_numerical)

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


class Seq2SeqNumericalizer(TransformerNumericalizer):
    
    def __init__(self, pretrained_tokenizer=None, config=None, max_generative_vocab=None, cache=None, fix_length=None):
        super().__init__(pretrained_tokenizer, config, max_generative_vocab, cache, fix_length)
        
    @property
    def pad_id(self):
        return self._tokenizer.pad_token_id
    
    def load(self, save_dir, config=None):
        raise NotImplementedError
    
    def save(self, save_dir):
        self._tokenizer.save_pretrained(save_dir)

    def build_vocab(self, vocab_fields, vocab_sets):
        raise NotImplementedError

    def encode_single(self, minibatch, decoder_vocab, max_length=-1):
        """
        minibatch: this method ignores the `mask` component of minibatch
        """
        assert isinstance(minibatch, list)
        batch_tokens = []
        for tokens, mask in minibatch:
            if len(tokens) == 0:
                batch_tokens.append(['']) # empty array makes tokenizer crash
            else:
                batch_tokens.append(tokens)
                
        encoded_batch = self._tokenizer.batch_encode_plus(batch_tokens, add_special_tokens=True, padding=False, return_attention_mask=False, is_split_into_words=True)
        numerical = encoded_batch['input_ids']
        length = [len(a) for a in encoded_batch['input_ids']]

        decoder_numerical = numerical

        return SequentialField(length=length, value=numerical, limited=decoder_numerical)

    def encode_pair(self, minibatch, decoder_vocab):
        # TODO
        raise NotImplementedError

    def reverse(self, batch, detokenize, field_name=None):
        _reversed = self._tokenizer.batch_decode(batch, skip_special_tokens=True)
        return _reversed

    def decode(self, tensor):
        return self._tokenizer.convert_ids_to_tokens(tensor)


class BartNumericalizer(Seq2SeqNumericalizer):
    
    def __init__(self, pretrained_tokenizer=None, config=None, max_generative_vocab=None, cache=None, fix_length=None):
        super().__init__(pretrained_tokenizer, config, max_generative_vocab, cache, fix_length)
        self.load(pretrained_tokenizer, config)
        
    def load(self, save_dir, config=None):
        self._tokenizer = BartTokenizer.from_pretrained(save_dir)
        self.decoder_vocab = DecoderVocabulary(self._tokenizer.decoder.values(), None,
                                               pad_token=self._tokenizer.pad_token, eos_token=self._tokenizer.eos_token)

class MBartNumericalizer(Seq2SeqNumericalizer):
    
    def __init__(self, pretrained_tokenizer=None, config=None, max_generative_vocab=None, cache=None, fix_length=None):
        super().__init__(pretrained_tokenizer, config, max_generative_vocab, cache, fix_length)
        self.load(pretrained_tokenizer, config)
    
    def load(self, save_dir, config=None):
        self._tokenizer = MBartTokenizer.from_pretrained(save_dir, config=config)
        vocabs = ['<pad>'] + [self._tokenizer.sp_model.id_to_piece(i) for i in
                              range(self._tokenizer.sp_model.get_piece_size())]
        self.decoder_vocab = DecoderVocabulary(vocabs, None, pad_token=self._tokenizer.pad_token,
                                               eos_token=self._tokenizer.eos_token)

class MT5Numericalizer(Seq2SeqNumericalizer):
    
    def __init__(self, pretrained_tokenizer=None, config=None, max_generative_vocab=None, cache=None, fix_length=None):
        super().__init__(pretrained_tokenizer, config, max_generative_vocab, cache, fix_length)
        self.load(pretrained_tokenizer, config)
        
    def load(self, save_dir, config=None):
        self._tokenizer = T5Tokenizer.from_pretrained(save_dir, config=config)
        vocabs = [self._tokenizer.sp_model.id_to_piece(i) for i in range(self._tokenizer.sp_model.get_piece_size())]
        self.decoder_vocab = DecoderVocabulary(vocabs, None, pad_token=self._tokenizer.pad_token,
                                               eos_token=self._tokenizer.eos_token)
