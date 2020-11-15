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
import random
import torch

from .numericalizer.sequential_field import SequentialField
from torch.nn.utils.rnn import pad_sequence


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


class NumericalizedExamples(NamedTuple):
    example_id: List[str]
    context: SequentialField
    question: SequentialField
    answer: SequentialField
    decoder_vocab: object
    
    @staticmethod
    def from_examples(examples, numericalizer, device=None, paired=False, max_pairs=None, groups=None,
                      append_question_to_context_too=False, override_question=None, override_context=None):
        assert all(isinstance(ex.example_id, str) for ex in examples)
        decoder_vocab = numericalizer.decoder_vocab.clone()
        max_context_len, max_question_len, max_answer_len = -1, -1, -1

        override_question_mask = None
        if override_question:
            override_question = override_question.split()
            override_question_mask = [True for _ in override_question]
            
        override_context_mask = None
        if override_context:
            override_context = override_context.split()
            override_context_mask = [True for _ in override_context]

        if paired:
            example_pairs = []
            
            # get all possible combinations of related example pairs
            for i in range(0, len(examples), groups):
                related_examples = [examples[j] for j in range(i, i+groups)]
                example_pairs.extend(itertools.product(related_examples, related_examples))
            # filter out pairs with same sentences
            example_pairs = [ex_pair for ex_pair in example_pairs if ex_pair[0] != ex_pair[1]]
            
            # shuffle example orders and select first max_pairs of them
            random.shuffle(example_pairs)
            example_pairs = example_pairs[:max_pairs]
            
            example_ids = [ex_a.example_id + '@' + ex_b.example_id for ex_a, ex_b in example_pairs]
            if override_question:
                question_inputs = [((override_question, override_question_mask),
                                    (override_question, override_question_mask))
                                   for _ in example_pairs]
            else:
                question_inputs = [((ex_a.question, ex_a.question_word_mask),
                                    (ex_b.question, ex_b.question_word_mask))
                                   for ex_a, ex_b in example_pairs]
                
            if append_question_to_context_too:
                context_inputs = [((ex_a.context_plus_question,
                                    ex_a.context_plus_question_word_mask),
                                    (ex_b.context_plus_question,
                                     ex_b.context_plus_question_word_mask))
                                  for ex_a, ex_b in example_pairs]
            elif override_context:
                context_inputs = [((override_context, override_context_mask),
                                    (override_context, override_context_mask))
                                   for _ in example_pairs]
            else:
                context_inputs = [((ex_a.context, ex_a.context_word_mask),
                                    (ex_b.context, ex_b.context_word_mask))
                                   for ex_a, ex_b in example_pairs]


            answer_inputs = [((ex_a.answer, ex_a.answer_word_mask), (ex_b.answer, ex_b.answer_word_mask))
                             for ex_a, ex_b in example_pairs]

            all_example_ids_pair = example_ids
            all_context_inputs_pair = numericalizer.encode_pair(context_inputs, decoder_vocab, device=device)
            all_question_inputs_pair = numericalizer.encode_pair(question_inputs, decoder_vocab, device=device)
            all_answer_inputs_pair = numericalizer.encode_pair(answer_inputs, decoder_vocab, device=device)

            max_context_len = all_context_inputs_pair.value.size(1)
            max_question_len = all_question_inputs_pair.value.size(1)
            max_answer_len = all_answer_inputs_pair.value.size(1)

        # process single examples
        example_ids = [ex.example_id for ex in examples]
        if override_question:
            question_inputs = [(override_question, override_question_mask) for _ in examples]
        else:
            question_inputs = [(ex.question, ex.question_word_mask) for ex in examples]

        if append_question_to_context_too:
            context_inputs = [((ex.context, ex.context_word_mask), (ex.question, ex.question_word_mask)) for ex in examples]
        elif override_context:
            context_inputs = [(override_context, override_context_mask) for _ in examples]
        else:
            context_inputs = [(ex.context, ex.context_word_mask) for ex in examples]
            
        answer_inputs = [(ex.answer, ex.answer_word_mask) for ex in examples]
        
        all_example_ids_single = example_ids
        if append_question_to_context_too:
            all_context_inputs_single = numericalizer.encode_pair(context_inputs, decoder_vocab, device=device)
        else:
            all_context_inputs_single = numericalizer.encode_single(context_inputs, decoder_vocab,
                                                                device=device, max_length=max_context_len-2)
        all_question_inputs_single = numericalizer.encode_single(question_inputs, decoder_vocab,
                                                                 device=device, max_length=max_question_len-2)
        all_answer_inputs_single = numericalizer.encode_single(answer_inputs, decoder_vocab,
                                                               device=device, max_length=max_answer_len-2)
    
        if paired:
            all_example_ids = all_example_ids_single + all_example_ids_pair
            all_context_inputs = SequentialField.from_tensors([all_context_inputs_single, all_context_inputs_pair])
            all_question_inputs = SequentialField.from_tensors([all_question_inputs_single, all_question_inputs_pair])
            all_answer_inputs = SequentialField.from_tensors([all_answer_inputs_single, all_answer_inputs_pair])
        else:
            all_example_ids = all_example_ids_single
            all_context_inputs = all_context_inputs_single
            all_question_inputs = all_question_inputs_single
            all_answer_inputs = all_answer_inputs_single
        ret =  NumericalizedExamples(all_example_ids,
                     all_context_inputs,
                     all_question_inputs,
                     all_answer_inputs,
                     decoder_vocab)
        return ret

    @staticmethod
    def collate_batches(batches):
        example_id = []
        context_values, context_lengths, context_limiteds, context_segments = [], [], [], []
        question_values, question_lengths, question_limiteds = [], [], []
        answer_values, answer_lengths, answer_limiteds = [], [], []
        decoder_vocab = None

        max_context_length = 0
        max_question_length = 0
        max_answer_length = 0
        min_context_length = 1000000
        for batch in batches:
            example_id.append(batch.example_id[0])
            context_values.append(batch.context.value)
            context_lengths.append(batch.context.length)
            context_limiteds.append(batch.context.limited)
            context_segments.append(batch.context.segments)
            max_context_length = max(max_context_length, batch.context.length)
            min_context_length = min(min_context_length, batch.context.length)

            question_values.append(batch.question.value)
            question_lengths.append(batch.question.length)
            question_limiteds.append(batch.question.limited)
            max_question_length = max(max_question_length, batch.question.length)

            answer_values.append(batch.answer.value)
            answer_lengths.append(batch.answer.length)
            answer_limiteds.append(batch.answer.limited)
            max_answer_length = max(max_answer_length, batch.answer.length)

            decoder_vocab = batch.decoder_vocab


        context_values = pad_sequence(context_values, padding_value=0, batch_first=True)
        context_limiteds = pad_sequence(context_limiteds, padding_value=0, batch_first=True)
        context_segments = pad_sequence(context_segments, padding_value=1, batch_first=True)
        context_lengths = torch.stack(context_lengths, dim=0)
        question_values = pad_sequence(question_values, padding_value=0, batch_first=True)
        question_limiteds = pad_sequence(question_limiteds, padding_value=0, batch_first=True)
        question_lengths = torch.stack(question_lengths, dim=0)
        answer_values = pad_sequence(answer_values, padding_value=0, batch_first=True)
        answer_limiteds = pad_sequence(answer_limiteds, padding_value=0, batch_first=True)
        answer_lengths = torch.stack(answer_lengths, dim=0)

        context = SequentialField(value=context_values,
                                  length=context_lengths,
                                  limited=context_limiteds,
                                  segments=context_segments)

        question = SequentialField(value=question_values,
                                   length=question_lengths,
                                   limited=question_limiteds)

        answer = SequentialField(value=answer_values,
                                 length=answer_lengths,
                                 limited=answer_limiteds)


        return NumericalizedExamples(example_id=example_id, context=context, question=question, answer=answer, decoder_vocab=decoder_vocab)