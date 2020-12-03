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
from typing import Callable

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
    context_plus_question: List[str]
    context_plus_question_word_mask: List[bool]

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
        
        # create context_plus_question field by appending context and question words and words_masks
        args.append(args[1] + args[3])
        args.append(args[2] + args[4])
        
        return Example(*args)


class NumericalizedExamples(NamedTuple):
    example_id: List[str]
    context: SequentialField
    question: SequentialField
    answer: SequentialField
    decoder_vocab: object
    device: torch.device
    padding_function: Callable
    
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
            all_context_inputs_pair = numericalizer.encode_pair(context_inputs, decoder_vocab)
            all_question_inputs_pair = numericalizer.encode_pair(question_inputs, decoder_vocab)
            all_answer_inputs_pair = numericalizer.encode_pair(answer_inputs, decoder_vocab)

            max_context_len = max(all_context_inputs_pair.length)
            max_question_len = max(all_question_inputs_pair.length)
            max_answer_len = max(all_answer_inputs_pair.length)

        # process single examples
        example_ids = [ex.example_id for ex in examples]
        if override_question:
            question_inputs = [(override_question, override_question_mask) for _ in examples]
        else:
            question_inputs = [(ex.question, ex.question_word_mask) for ex in examples]

        if append_question_to_context_too:
            context_inputs = [(ex.context_plus_question, ex.context_plus_question_word_mask) for ex in examples]
        elif override_context:
            context_inputs = [(override_context, override_context_mask) for _ in examples]
        else:
            context_inputs = [(ex.context, ex.context_word_mask) for ex in examples]
            
        answer_inputs = [(ex.answer, ex.answer_word_mask) for ex in examples]
        
        all_example_ids_single = example_ids
        all_context_inputs_single = numericalizer.encode_single(context_inputs, decoder_vocab,
                                                                max_length=max_context_len-2)
        all_question_inputs_single = numericalizer.encode_single(question_inputs, decoder_vocab,
                                                                 max_length=max_question_len-2)
        all_answer_inputs_single = numericalizer.encode_single(answer_inputs, decoder_vocab,
                                                               max_length=max_answer_len-2)
    
        if paired:
            all_example_ids = all_example_ids_single + all_example_ids_pair
            all_context_inputs = SequentialField.merge([all_context_inputs_single, all_context_inputs_pair])
            all_question_inputs = SequentialField.merge([all_question_inputs_single, all_question_inputs_pair])
            all_answer_inputs = SequentialField.merge([all_answer_inputs_single, all_answer_inputs_pair])
        else:
            all_example_ids = all_example_ids_single
            all_context_inputs = all_context_inputs_single
            all_question_inputs = all_question_inputs_single
            all_answer_inputs = all_answer_inputs_single
        return NumericalizedExamples(all_example_ids,
                     all_context_inputs,
                     all_question_inputs,
                     all_answer_inputs,
                     decoder_vocab,
                     device, padding_function=numericalizer.pad)

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


        return NumericalizedExamples(example_id=example_id, context=context, question=question, answer=answer, decoder_vocab=decoder_vocab, device=None, padding_function=padding_function)