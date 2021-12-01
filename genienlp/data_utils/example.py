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

import unicodedata
from typing import Iterable, List, NamedTuple, Union

import torch


def identity(x, **kw):
    return x


class SequentialField(NamedTuple):
    value: Union[torch.tensor, List[int]]
    length: Union[torch.tensor, int]
    limited: Union[torch.tensor, List[int]]
    feature: Union[torch.tensor, List[List[int]], None]


VALID_ENTITY_ATTRIBUTES = ('type_id', 'type_prob', 'qid')


# Entity is defined per token
# Each attribute contains a list of possible values for that entity
class Entity(object):
    def __init__(
        self,
        type_id: List[int] = None,
        type_prob: List[float] = None,
        qid: List[int] = None,
    ):
        self.type_id = type_id
        self.type_prob = type_prob
        self.qid = qid

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def flatten(self):
        result = []
        for field in VALID_ENTITY_ATTRIBUTES:
            field_val = getattr(self, field)
            if field_val:
                result += field_val
        return result

    @staticmethod
    def get_pad_entity(max_features_size):
        pad_feature = Entity()
        for i, field in enumerate(VALID_ENTITY_ATTRIBUTES):
            setattr(pad_feature, field, [0] * max_features_size)
        return pad_feature


class Example(object):
    """
    Contains all fields of a train/dev/test example in text form, alongside their NED features in embedding_id form (`*_feature`)
    """

    def __init__(
        self,
        example_id: str,
        context: str,
        context_feature: List[Entity],
        question: str,
        question_feature: List[Entity],
        answer: str,
    ):

        self.example_id = example_id
        self.context = context
        self.context_feature = context_feature
        self.question = question
        self.question_feature = question_feature
        self.answer = answer

    @staticmethod
    def from_raw(example_id: str, context: str, question: str, answer: str, preprocess=identity, lower=False):
        args = [example_id]
        answer = unicodedata.normalize('NFD', answer)

        for argname, arg in (('context', context), ('question', question), ('answer', answer)):
            arg = unicodedata.normalize('NFD', arg)
            if lower:
                arg = arg.lower()

            sentence = preprocess(arg.rstrip('\n'), field_name=argname, answer=answer, example_id=example_id)

            args.append(sentence)

            if argname != 'answer':
                # we use a placeholder for features here
                # the features will be produced and overridden via bootleg or database
                args.append([])

        return Example(*args)


class NumericalizedExamples(NamedTuple):
    """
    Contains a batch of numericalized (i.e. tokenized and converted to token ids) examples, potentially of size 1
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
            pad_feature = [Entity.get_pad_entity(args.max_features_size)]

        # we keep the result of concatenation of question and context fields in these arrays temporarily. The numericalized versions will live on in self.context
        all_context_plus_questions = []
        all_context_plus_question_features = []

        for ex in examples:
            if not len(ex.question):
                context_plus_question = ex.context
            elif not len(ex.context):
                context_plus_question = ex.question
            else:
                context_plus_question = ex.context + sep_token + ex.question

            all_context_plus_questions.append(context_plus_question)

            # concatenate question and context features with a separator, but no need for a separator if there are no features to begin with
            context_plus_question_feature = (
                ex.context_feature + pad_feature + ex.question_feature
                if len(ex.question_feature) + len(ex.context_feature) > 0
                else []
            )
            all_context_plus_question_features.append(context_plus_question_feature)

        if args.do_ned and args.add_entities_to_text == 'off':
            features = all_context_plus_question_features
        else:
            # features are already processed and added to input as text
            features = None

        tokenized_contexts = numericalizer.encode_batch(all_context_plus_questions, field_name='context', features=features)

        # TODO remove double attempts at context tokenization
        if getattr(examples, 'is_classification', False):
            tokenized_answers = numericalizer.process_classification_labels(
                all_context_plus_questions, [ex.answer for ex in examples]
            )
        elif getattr(examples, 'is_sequence_classification', False):
            # align labels
            answers = [
                [
                    int(ex.answer),
                ]
                for ex in examples
            ]

            batch_decoder_numerical = []
            if numericalizer.decoder_vocab:
                for i in range(len(answers)):
                    batch_decoder_numerical.append(numericalizer.decoder_vocab.encode(answers[i]))
            else:
                batch_decoder_numerical = [[]] * len(answers)

            tokenized_answers = []
            for i in range(len(answers)):
                tokenized_answers.append(
                    SequentialField(
                        value=answers[i],
                        length=len(answers[i]),
                        limited=batch_decoder_numerical[i],
                        feature=None,
                    )
                )

        else:
            tokenized_answers = numericalizer.encode_batch([ex.answer for ex in examples], field_name='answer')

        for i in range(len(examples)):
            numericalized_examples.append(
                NumericalizedExamples([examples[i].example_id], tokenized_contexts[i], tokenized_answers[i])
            )
        return numericalized_examples

    @staticmethod
    def collate_batches(batches: Iterable['NumericalizedExamples'], numericalizer, device):
        example_id = []

        context_values, context_lengths, context_limiteds, context_features = [], [], [], []
        answer_values, answer_lengths, answer_limiteds = [], [], []

        for batch in batches:
            example_id.append(batch.example_id[0])
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

        context = SequentialField(
            value=context_values,
            length=context_lengths,
            limited=context_limiteds,
            feature=context_features,
        )

        answer = SequentialField(value=answer_values, length=answer_lengths, limited=answer_limiteds, feature=None)

        return NumericalizedExamples(example_id=example_id, context=context, answer=answer)
