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
from typing import NamedTuple, List, Union, Iterable
import torch


def identity(x, **kw):
    return x


class SequentialField(NamedTuple):
    value: Union[torch.tensor, List[int]]
    length: Union[torch.tensor, int]
    limited: Union[torch.tensor, List[int]]


class Example(NamedTuple):
    example_id: str
    context: str
    question: str
    answer: str
    context_plus_question: List[str]

    @staticmethod
    def from_raw(example_id: str, context: str, question: str, answer: str, preprocess = identity, lower=False):
        args = [example_id]

        for argname, arg in (('context', context), ('question', question), ('answer', answer)):
            arg = unicodedata.normalize('NFD', arg)
            field = preprocess(arg.rstrip('\n'), field_name=argname).strip()
            if lower:
                field = field.lower()
            args.append(field)

        # create context_plus_question field by appending context and question words
        args.append(args[1] + ' ' + args[2])

        return Example(*args)


class NumericalizedExamples(NamedTuple):
    example_id: List[str]
    context: SequentialField
    answer: SequentialField
    
    @staticmethod
    def from_examples(examples, numericalizer):
        assert all(isinstance(ex.example_id, str) for ex in examples)
        numericalized_examples = []

        tokenized_contexts = numericalizer.encode_batch([ex.context_plus_question for ex in examples])
        tokenized_answers = numericalizer.encode_batch([ex.answer for ex in examples])
        for i in range(len(examples)):
            numericalized_examples.append(NumericalizedExamples([examples[i].example_id],
                                        tokenized_contexts[i],
                                        tokenized_answers[i]))
        return numericalized_examples

    @staticmethod
    def collate_batches(batches : Iterable['NumericalizedExamples'], numericalizer, device):
        example_id = []
        context_values, context_lengths, context_limiteds = [], [], []
        answer_values, answer_lengths, answer_limiteds = [], [], []

        for batch in batches:
            example_id.append(batch.example_id[0])
            context_values.append(torch.tensor(batch.context.value, device=device))
            context_lengths.append(torch.tensor(batch.context.length, device=device))
            context_limiteds.append(torch.tensor(batch.context.limited, device=device))

            answer_values.append(torch.tensor(batch.answer.value, device=device))
            answer_lengths.append(torch.tensor(batch.answer.length, device=device))
            answer_limiteds.append(torch.tensor(batch.answer.limited, device=device))

        context_values = numericalizer.pad(context_values, pad_id=numericalizer.pad_id)
        context_limiteds = numericalizer.pad(context_limiteds, pad_id=numericalizer.decoder_pad_id)
        context_lengths = torch.stack(context_lengths, dim=0)
        answer_values = numericalizer.pad(answer_values, pad_id=numericalizer.pad_id)
        answer_limiteds = numericalizer.pad(answer_limiteds, pad_id=numericalizer.decoder_pad_id)
        answer_lengths = torch.stack(answer_lengths, dim=0)

        context = SequentialField(value=context_values,
                                  length=context_lengths,
                                  limited=context_limiteds)

        answer = SequentialField(value=answer_values,
                                 length=answer_lengths,
                                 limited=answer_limiteds)


        return NumericalizedExamples(example_id=example_id,
                                     context=context,
                                     answer=answer)