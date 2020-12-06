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

from typing import NamedTuple, List, Union
import torch
from typing import Callable

from .numericalizer.sequential_field import SequentialField


def identity(x, **kw):
    return x


class Example(NamedTuple):
    example_id: str
    context: str
    question: str
    answer: str
    context_plus_question: List[str]

    vocab_fields = ['context', 'question', 'answer']

    @staticmethod
    def from_raw(example_id: str, context: str, question: str, answer: str, preprocess = identity, lower=False):
        args = [example_id]

        for argname, arg in (('context', context), ('question', question), ('answer', answer)):
            field = preprocess(arg.rstrip('\n'), field_name=argname)
            if lower:
                field = field.lower()
            args.append(field)

        # create context_plus_question field by appending context and question words
        args.append(args[1] + ' ' + args[2])

        return Example(*args)


class NumericalizedExamples(NamedTuple):
    example_id: List[str]
    context: SequentialField
    question: SequentialField
    answer: SequentialField
    decoder_vocab: object
    device: Union[torch.device, None]
    padding_function: Callable
    
    @staticmethod
    def from_examples(examples, numericalizer, device):
        assert all(isinstance(ex.example_id, str) for ex in examples)
        decoder_vocab = numericalizer.decoder_vocab.clone()
        max_context_len, max_question_len, max_answer_len = -1, -1, -1

        for ex in examples:
            yield NumericalizedExamples(ex.example_id,
                                        numericalizer.encode_single(ex.context, decoder_vocab,
                                                                    max_length=max_context_len - 2),
                                        numericalizer.encode_single(ex.question, decoder_vocab,
                                                                    max_length=max_context_len - 2),
                                        numericalizer.encode_single(ex.answer, decoder_vocab,
                                                                    max_length=max_answer_len - 2),
                                        decoder_vocab, device=device, padding_function=numericalizer.pad)

    @staticmethod
    def collate_batches(batches):
        example_id = []
        context_values, context_lengths, context_limiteds = [], [], []
        question_values, question_lengths, question_limiteds = [], [], []
        answer_values, answer_lengths, answer_limiteds = [], [], []
        decoder_vocab = None

        for batch in batches:
            example_id.append(batch.example_id[0])
            context_values.append(torch.tensor(batch.context.value, device=batch.device))
            context_lengths.append(torch.tensor(batch.context.length, device=batch.device))
            context_limiteds.append(torch.tensor(batch.context.limited, device=batch.device))

            question_values.append(torch.tensor(batch.question.value, device=batch.device))
            question_lengths.append(torch.tensor(batch.question.length, device=batch.device))
            question_limiteds.append(torch.tensor(batch.question.limited, device=batch.device))

            answer_values.append(torch.tensor(batch.answer.value, device=batch.device))
            answer_lengths.append(torch.tensor(batch.answer.length, device=batch.device))
            answer_limiteds.append(torch.tensor(batch.answer.limited, device=batch.device))

            decoder_vocab = batch.decoder_vocab
            padding_function = batch.padding_function

        context_values = padding_function(context_values)
        context_limiteds = padding_function(context_limiteds)
        context_lengths = torch.stack(context_lengths, dim=0)
        question_values = padding_function(question_values)
        question_limiteds = padding_function(question_limiteds)
        question_lengths = torch.stack(question_lengths, dim=0)
        answer_values = padding_function(answer_values)
        answer_limiteds = padding_function(answer_limiteds)
        answer_lengths = torch.stack(answer_lengths, dim=0)

        context = SequentialField(value=context_values,
                                  length=context_lengths,
                                  limited=context_limiteds)

        question = SequentialField(value=question_values,
                                   length=question_lengths,
                                   limited=question_limiteds)

        answer = SequentialField(value=answer_values,
                                 length=answer_lengths,
                                 limited=answer_limiteds)


        return NumericalizedExamples(example_id=example_id,
                                     context=context,
                                     question=question,
                                     answer=answer,
                                     decoder_vocab=decoder_vocab,
                                     device=None,
                                     padding_function=padding_function)