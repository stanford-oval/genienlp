#
# Copyright (c) 2018, Salesforce, Inc.
#                     The Board of Trustees of the Leland Stanford Junior University
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

import torch
from typing import NamedTuple, List


class SequentialField(NamedTuple):
    value : torch.tensor
    length : torch.tensor
    limited : torch.tensor
    tokens : List[str]


class Example(NamedTuple):
    example_id : str
    context : List[str]
    question : List[str]
    answer : List[str]

    vocab_fields = ['context', 'question', 'answer']

    @staticmethod
    def from_raw(example_id : str, context : str, question : str, answer : str, tokenize, lower=False):
        args = [example_id]
        for argname, arg in (('context', context), ('question', question), ('answer', answer)):
            new_arg = tokenize(arg.rstrip('\n'), field_name=argname)
            if lower:
                new_arg = [word.lower() for word in new_arg]
            args.append(new_arg)
        return Example(*args)


class Batch(NamedTuple):
    example_id : List[str]
    context : SequentialField
    question : SequentialField
    answer : SequentialField
    decoder_vocab : object

    @staticmethod
    def from_examples(examples, numericalizer, device=None):
        assert all(isinstance(ex.example_id, str) for ex in examples)
        example_ids = [ex.example_id for ex in examples]
        context_input = [ex.context for ex in examples]
        question_input = [ex.question for ex in examples]
        answer_input = [ex.answer for ex in examples]
        decoder_vocab = numericalizer.decoder_vocab.clone()

        return Batch(example_ids,
                     numericalizer.encode(context_input, decoder_vocab, device=device),
                     numericalizer.encode(question_input, decoder_vocab, device=device),
                     numericalizer.encode(answer_input, decoder_vocab, device=device),
                     decoder_vocab)


