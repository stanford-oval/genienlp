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

from typing import NamedTuple, List
import itertools

from .numericalizer.sequential_field import SequentialField


class Example(NamedTuple):
    example_id: str

    # for each field in the example, we store the tokenized sentence, and a boolean mask
    # indicating whether the token is a real word (subject to word-piece tokenization)
    # or it should be treated as an opaque symbol
    context: List[str]
    context_word_mask: List[bool]
    question: List[str]
    question_word_mask: List[bool]
    answer: List[str]
    answer_word_mask: List[bool]

    vocab_fields = ['context', 'question', 'answer']

    @staticmethod
    def from_raw(example_id: str, context: str, question: str, answer: str, tokenize, lower=False):
        args = [example_id]
        for argname, arg in (('context', context), ('question', question), ('answer', answer)):
            words, mask = tokenize(arg.rstrip('\n'), field_name=argname)
            if mask is None:
                mask = [True for _ in words]
            if lower:
                words = [word.lower() for word in words]
            args.append(words)
            args.append(mask)
        return Example(*args)


class Batch(NamedTuple):
    example_id: List[str]
    context: SequentialField
    question: SequentialField
    answer: SequentialField
    decoder_vocab: object

    @staticmethod
    def from_examples_single(examples, numericalizer, device=None):
        assert all(isinstance(ex.example_id, str) for ex in examples)
        example_ids = [ex.example_id for ex in examples]
        context_input = [(ex.context, ex.context_word_mask) for ex in examples]
        question_input = [(ex.question, ex.question_word_mask) for ex in examples]
        answer_input = [(ex.answer, ex.answer_word_mask) for ex in examples]
        decoder_vocab = numericalizer.decoder_vocab.clone()

        return Batch(example_ids,
                     numericalizer.encode(context_input, decoder_vocab, device=device),
                     numericalizer.encode(question_input, decoder_vocab, device=device),
                     numericalizer.encode(answer_input, decoder_vocab, device=device),
                     decoder_vocab)
    
    @staticmethod
    def from_examples(examples, numericalizer, device=None, paired=False, groups=None):
        assert all(isinstance(ex.example_id, str) for ex in examples)
        
        decoder_vocab = numericalizer.decoder_vocab.clone()

        # process single examples
        example_ids = [ex.example_id for ex in examples]
        context_inputs = [(ex.context, ex.context_word_mask) for ex in examples]
        question_inputs = [(ex.question, ex.question_word_mask) for ex in examples]
        answer_inputs = [(ex.answer, ex.answer_word_mask) for ex in examples]
        
        all_example_ids = example_ids
        all_context_inputs = numericalizer.encode_single(context_inputs, decoder_vocab, device=device)
        all_question_inputs = numericalizer.encode_single(question_inputs, decoder_vocab, device=device)
        all_answer_inputs = numericalizer.encode_single(answer_inputs, decoder_vocab, device=device)
        
        
        if paired:
            example_pairs = []
            # get all possible combinations of related example pairs
            # TODO filter out pairs of same sentences
            for i in range(0, len(examples), groups):
                related_examples = [examples[j] for j in range(i, i+groups)]
                example_pairs.extend(itertools.product(related_examples, related_examples))

            example_ids = [ex_a.example_id + '@' + ex_b.example_id for ex_a, ex_b in example_pairs]
            context_inputs = [((ex_a.context, ex_a.context_word_mask), (ex_b.context, ex_b.context_word_mask)) for ex_a, ex_b in example_pairs]
            question_inputs = [((ex_a.question, ex_a.question_word_mask), (ex_b.question, ex_b.question_word_mask)) for ex_a, ex_b in example_pairs]
            answer_inputs = [((ex_a.answer, ex_a.answer_word_mask), (ex_b.answer, ex_b.answer_word_mask)) for ex_a, ex_b in example_pairs]
        
            all_example_ids.extend(example_ids)
            all_context_inputs.extend(numericalizer.encode_pair(context_inputs, decoder_vocab, device=device))
            all_question_inputs.extend(numericalizer.encode_pair(question_inputs, decoder_vocab, device=device))
            all_answer_inputs.extend(numericalizer.encode_pair(answer_inputs, decoder_vocab, device=device))
        

        return Batch(all_example_ids,
                     all_context_inputs,
                     all_question_inputs,
                     all_answer_inputs,
                     decoder_vocab)
