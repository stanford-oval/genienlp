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




class Example(object):
    """
    Contains all fields of a train/dev/test example in text form
    """

    def __init__(
        self,
        example_id: str,
        context: str,
        question: str,
        answer: str,
    ):

        self.example_id = example_id
        self.context = context
        self.question = question
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

        sep_token = ' ' + numericalizer.sep_token + ' '

        # we keep the result of concatenation of question and context fields in these arrays temporarily. The numericalized versions will live on in self.context
        all_context_plus_questions = []

        for ex in examples:
            if not len(ex.question):
                context_plus_question = ex.context
            elif not len(ex.context):
                context_plus_question = ex.question
            else:
                context_plus_question = ex.context + sep_token + ex.question

            all_context_plus_questions.append(context_plus_question)

        tokenized_contexts = numericalizer.encode_batch(all_context_plus_questions, field_name='context')
        tokenized_answers = numericalizer.encode_batch([ex.answer for ex in examples], field_name='answer')

        for i in range(len(examples)):
            numericalized_examples.append(
                NumericalizedExamples([examples[i].example_id], tokenized_contexts[i], tokenized_answers[i])
            )
        return numericalized_examples

    @staticmethod
    def collate_batches(batches: Iterable['NumericalizedExamples'], numericalizer):
        example_id = []

        context_values, context_lengths, context_limiteds = [], [], []
        answer_values, answer_lengths, answer_limiteds = [], [], []

        for batch in batches:
            example_id.append(batch.example_id[0])
            context_values.append(torch.tensor(batch.context.value))
            context_lengths.append(torch.tensor(batch.context.length))
            context_limiteds.append(torch.tensor(batch.context.limited))

            answer_values.append(torch.tensor(batch.answer.value))
            answer_lengths.append(torch.tensor(batch.answer.length))
            answer_limiteds.append(torch.tensor(batch.answer.limited))

        context_values = numericalizer.pad(context_values, pad_id=numericalizer.pad_id)
        context_limiteds = numericalizer.pad(context_limiteds, pad_id=numericalizer.decoder_pad_id)
        context_lengths = torch.stack(context_lengths, dim=0)


        answer_values = numericalizer.pad(answer_values, pad_id=numericalizer.pad_id)
        answer_limiteds = numericalizer.pad(answer_limiteds, pad_id=numericalizer.decoder_pad_id)
        answer_lengths = torch.stack(answer_lengths, dim=0)

        context = SequentialField(
            value=context_values,
            length=context_lengths,
            limited=context_limiteds,
        )

        answer = SequentialField(value=answer_values, length=answer_lengths, limited=answer_limiteds)

        return NumericalizedExamples(example_id=example_id, context=context, answer=answer)
