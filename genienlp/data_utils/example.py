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
    def from_raw(example_id: str, context: str, question: str, answer: str):
        args = [example_id]
        answer = unicodedata.normalize('NFD', answer)

        for arg in (context, question, answer):
            arg = unicodedata.normalize('NFD', arg)

            sentence = arg.rstrip('\n')
            args.append(sentence)

        return Example(*args)


class ExampleBatch(NamedTuple):
    example_id: List[str]
    context_string: Union[str, List[str]]
    answer_string: Union[str, List[str]]

    @staticmethod
    def example_list_to_batch(examples: Iterable[Example]):
        assert all(isinstance(ex.example_id, str) for ex in examples)

        return ExampleBatch(
            example_id=[e.example_id for e in examples],
            context_string=[e.context for e in examples],
            answer_string=[e.answer for e in examples],
        )
