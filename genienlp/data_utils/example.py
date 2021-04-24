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

from typing import NamedTuple, List, Union, Iterable
import unicodedata
import torch
from dataclasses import dataclass

def identity(x, **kw):
    return x, [], x


class SequentialField(NamedTuple):
    value: Union[torch.tensor, List[int]]
    length: Union[torch.tensor, int]
    limited: Union[torch.tensor, List[int]]
    feature: Union[torch.tensor, List[List[int]], None]


# Feature is defined per token
# Each field contains a list of possible values for that feature
@dataclass
class Feature:
    type_id: List[int] = None
    type_prob: List[float] = None

    def __mul__(self, n):
        return [self for _ in range(n)]

    def flatten(self):
        result = []
        for field in VALID_FEATURE_FIELDS:
            field_val = getattr(self, field)
            if field_val:
                result += field_val
        return result

VALID_FEATURE_FIELDS = tuple(Feature.__annotations__.keys())

def get_pad_feature(feature_fields, ned_features_default_val, ned_features_size):
    # return None if not using NED
    pad_feature = None
    if len(feature_fields):
        pad_feature = Feature()
        for i, field in enumerate(feature_fields):
            assert field in VALID_FEATURE_FIELDS
            setattr(pad_feature, field, [ned_features_default_val[i]] * ned_features_size[i])
    return pad_feature


class Example(object):
    """
    Contains all fields of a train/dev/test example in text form, alongside their NED features
    both in text form (`*_plus_types` fields) and in embedding_id form (`*_feature`)
    """

    def __init__(self,
                 example_id: str,
                 context: str,
                 context_feature: List[Feature],
                 context_plus_types: str,
                 question: str,
                 question_feature: List[Feature],
                 question_plus_types: str,
                 answer: str):

        self.example_id = example_id
        self.context = context
        self.context_feature = context_feature
        self.context_plus_types = context_plus_types
        self.question = question
        self.question_feature = question_feature
        self.question_plus_types = question_plus_types
        self.answer = answer


    @staticmethod
    def from_raw(example_id: str, context: str, question: str, answer: str, preprocess=identity, lower=False):
        args = [example_id]
        answer = unicodedata.normalize('NFD', answer)
        
        for argname, arg in (('context', context), ('question', question), ('answer', answer)):
            arg = unicodedata.normalize('NFD', arg)
            if lower:
                arg = arg.lower()
                
            sentence, features, sentence_plus_types = preprocess(arg.rstrip('\n'), field_name=argname, answer=answer)

            args.append(sentence)
    
            if argname != 'answer':
                args.append(features)

            if argname == 'context':
                context_plus_types = sentence_plus_types
                args.append(context_plus_types)
            elif argname == 'question':
                question_plus_types = sentence_plus_types
                args.append(question_plus_types)
        
        return Example(*args)


class NumericalizedExamples(NamedTuple):
    """
    Conatains a batch of numericalized (i.e. tokenized and converted to token ids) examples, potentially of size 1
    """
    example_id: List[str]
    context: SequentialField
    answer: SequentialField
    
    @staticmethod
    def from_examples(examples: Iterable[Example], numericalizer):
        assert all(isinstance(ex.example_id, str) for ex in examples)
        numericalized_examples = []
        args = numericalizer.args

        if args.no_separator:
            sep_token = ' '
            pad_feature = []
        else:
            sep_token = ' ' + numericalizer.sep_token + ' '
            pad_feature = [get_pad_feature(args.ned_features, args.ned_features_default_val, args.ned_features_size)]

        # we keep the result of concatenation of question and context fields in these arrays temporarily. The numericalized versions will live on in self.context
        all_context_plus_questions = []
        all_context_plus_question_with_types = []
        all_context_plus_question_features = []
        
        for ex in examples:
            # create context_plus_question fields by concatenating context and question fields
            # if question is empty, don't append anything
            context_plus_question = ex.context + sep_token + ex.question if len(ex.question) else ex.context
            all_context_plus_questions.append(context_plus_question)
            context_plus_question_with_types = ex.context_plus_types + sep_token + ex.question_plus_types if len(ex.question_plus_types) else ex.context_plus_types
            all_context_plus_question_with_types.append(context_plus_question_with_types)
            # concatenate question and context features with a separator, but no need for a separator if there are no features to begin with
            context_plus_question_feature = ex.context_feature + pad_feature + ex.question_feature if len(ex.question_feature) + len(ex.context_feature) > 0 else []
            all_context_plus_question_features.append(context_plus_question_feature)
        
        if args.add_types_to_text == 'no':
            tokenized_contexts = numericalizer.encode_batch(
                    all_context_plus_questions,
                    field_name='context',
                    features=[a for a in all_context_plus_question_features if a]
            )
        else:
            tokenized_contexts = numericalizer.encode_batch(
                all_context_plus_question_with_types,
                field_name='context'
            )
        
        #TODO remove double attempts at context tokenization
        if getattr(examples, 'is_classification', False):
            assert args.add_types_to_text != 'insert'
            tokenized_answers = numericalizer.process_classification_labels(all_context_plus_questions, all_context_plus_question_with_types, [ex.answer for ex in examples])
        else:
            tokenized_answers = numericalizer.encode_batch([ex.answer for ex in examples], field_name='answer')
        
        for i in range(len(examples)):
            numericalized_examples.append(NumericalizedExamples([examples[i].example_id], tokenized_contexts[i], tokenized_answers[i]))
        return numericalized_examples

    @staticmethod
    def collate_batches(batches: Iterable['NumericalizedExamples'], numericalizer, device):
        example_id = []

        context_values, context_lengths, context_limiteds, context_features = [], [], [], []
        answer_values, answer_lengths, answer_limiteds = [], [], []

        for batch in batches:
            example_id.append(batch.example_id[0])
            
            # apply subword dropout on context
            if numericalizer.args.csp_dropout > 0.0 and numericalizer.semi_colon_id in batch.context.value:
                #TODOS mask complete words instead of subwords
                value = batch.context.value
                semi_colon_idx = value.index(numericalizer.semi_colon_id)
                bernoulli = torch.bernoulli(torch.ones([1, semi_colon_idx]) * (1 - numericalizer.args.csp_dropout))
                bernoulli = torch.cat([bernoulli, torch.ones([1, len(value)-semi_colon_idx])], dim=1)
                new_value = torch.where(bernoulli==0, numericalizer.mask_id, torch.tensor(value)).tolist()[0]
                context_values.append(torch.tensor(new_value, device=device))
                    
            else:
                context_values.append(torch.tensor(batch.context.value, device=device))
            context_lengths.append(torch.tensor(batch.context.length, device=device))
            context_limiteds.append(torch.tensor(batch.context.limited, device=device))
            if batch.context.feature:
                context_features.append(torch.tensor(batch.context.feature, device=device))

            answer_values.append(torch.tensor(batch.answer.value, device=device))
            answer_lengths.append(torch.tensor(batch.answer.length, device=device))
            answer_limiteds.append(torch.tensor(batch.answer.limited, device=device))

        context_values = numericalizer.pad(context_values, pad_id=numericalizer.pad_id)
        context_limiteds = numericalizer.pad(context_limiteds, pad_id=numericalizer.decoder_pad_id)
        context_lengths = torch.stack(context_lengths, dim=0)
        
        if context_features:
            context_features = numericalizer.pad(context_features, pad_id=numericalizer.args.db_unk_id)
        
        answer_values = numericalizer.pad(answer_values, pad_id=numericalizer.pad_id)
        answer_limiteds = numericalizer.pad(answer_limiteds, pad_id=numericalizer.decoder_pad_id)
        answer_lengths = torch.stack(answer_lengths, dim=0)

        context = SequentialField(value=context_values,
                                  length=context_lengths,
                                  limited=context_limiteds,
                                  feature=context_features)


        answer = SequentialField(value=answer_values,
                                 length=answer_lengths,
                                 limited=answer_limiteds,
                                 feature=None)


        return NumericalizedExamples(example_id=example_id,
                                     context=context,
                                     answer=answer)
