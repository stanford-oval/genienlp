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

import json
import logging
import os
from typing import Iterable

import ujson
from datasets import load_dataset

from ..data_utils.example import Example, NumericalizedExamples
from .base_dataset import Dataset, Split, interleave_keys

logger = logging.getLogger(__name__)


def make_example_id(dataset, example_id):
    return dataset.name + '/' + str(example_id)


# sort_key funcs
def context_answer_len(ex: NumericalizedExamples):
    return interleave_keys(ex.context.length, ex.answer.length)


def context_question_len(ex: NumericalizedExamples):
    return ex.context.length  # question is already appended to context


def input_then_output_len(ex: NumericalizedExamples):
    """
    sort by input length, break ties by output length
    """
    return (context_question_len(ex), answer_len(ex))


def answer_len(ex: NumericalizedExamples):
    return ex.answer.length


def id_value(ex):
    id_ = ex.example_id.rsplit('/', 1)
    id_ = id_[0] if len(id_) == 1 else id_[1]
    return id_


# batch_size functions; batch size is calculated after pad tokens are added
def input_tokens_fn(batch: Iterable[NumericalizedExamples]):
    return max([context_question_len(e) for e in batch]) * len(batch)


def all_tokens_fn(batch: Iterable[NumericalizedExamples]):
    return (max([context_question_len(e) for e in batch]) + max([answer_len(e) for e in batch])) * len(batch)


def default_batch_fn(batch: Iterable[NumericalizedExamples]):
    return len(batch)


class CQA(Dataset):
    def __init__(self, examples, sort_key_fn=input_then_output_len, batch_size_fn=all_tokens_fn, groups=None, **kwargs):
        self.sort_key_fn = sort_key_fn
        self.batch_size_fn = batch_size_fn
        self.groups = groups
        super().__init__(examples, **kwargs)


class JSON(CQA):
    name = 'json'

    def __init__(self, path, subsample=None, lower=False, **kwargs):

        examples = []
        with open(os.path.expanduser(path)) as f:
            lines = f.readlines()
            for line in lines:
                ex = json.loads(line)
                context, question, answer = ex['context'], ex['question'], ex['answer']
                examples.append(Example.from_raw(make_example_id(self, len(examples)), context, question, answer, lower=lower))
                if subsample is not None and len(examples) >= subsample:
                    break

        super(JSON, self).__init__(examples, **kwargs)

    @classmethod
    def splits(cls, root='.data', name=None, train='train', validation='val', test='test', **kwargs):
        path = os.path.join(root, name)

        train_data = None if train is None else cls(os.path.join(path, 'train.jsonl'), **kwargs)
        validation_data = None if validation is None else cls(os.path.join(path, 'val.jsonl'), **kwargs)
        test_data = None if test is None else cls(os.path.join(path, 'test.jsonl'), **kwargs)

        aux_data = None
        do_curriculum = kwargs.get('curriculum', False)
        if do_curriculum:
            kwargs.pop('curriculum')
            aux_data = cls(os.path.join(path, 'aux.jsonl'), **kwargs)

        return Split(
            train=None if train is None else train_data,
            eval=None if validation is None else validation_data,
            test=None if test is None else test_data,
            aux=None if do_curriculum is None else aux_data,
        )


class CrossNERDataset(CQA):
    is_classification = True

    def __init__(self, data, *, make_example, **kwargs):

        subsample = kwargs.pop('subsample')
        domain = kwargs.pop('domain')
        examples = []

        example_id, tokens, labels = 0, [], []
        for i, line in enumerate(data):
            line = line.strip()
            if line == "":
                # reached end of this example
                if len(tokens):
                    examples.append(make_example([example_id, tokens, labels], domain))
                tokens, labels = [], []
                example_id += 1
            else:
                splits = line.split("\t")
                tokens.append(splits[0])
                labels.append(splits[1])

            if subsample is not None and len(examples) >= subsample:
                break

        super().__init__(examples, **kwargs)

    @classmethod
    def return_splits(cls, path='.data', train='train', validation='dev', test='test', **kwargs):

        crossner_domains = kwargs.pop('crossner_domains')

        all_train_data = []
        all_validation_data = []
        all_test_data = []
        for domain in crossner_domains:
            # download datasets and cache them
            train_data, validation_data, test_data = None, None, None
            train_path, validation_path, test_path = None, None, None
            if train:
                train_path = os.path.join(path, domain, 'train.txt')
                with open(train_path, "r") as fin:
                    train_data = fin.readlines()
            if validation:
                validation_path = os.path.join(path, domain, f'{validation}.txt')
                with open(validation_path, "r") as fin:
                    validation_data = fin.readlines()
            if test:
                test_path = os.path.join(path, domain, 'test.txt')
                with open(test_path, "r") as fin:
                    test_data = fin.readlines()

            # Uncomment for debugging
            # if True:
            #     if validation:
            #         validation_path = os.path.join(path, domain, 'train.txt')
            #         with open(validation_path, "r") as fin:
            #             validation_data = fin.readlines()
            #     if test:
            #         test_path = os.path.join(path, domain, 'train.txt')
            #         with open(test_path, "r") as fin:
            #             test_data = fin.readlines()

            kwargs['domain'] = domain

            train_data = None if train is None else cls(train_data, **kwargs)
            validation_data = None if validation is None else cls(validation_data, **kwargs)
            test_data = None if test is None else cls(test_data, **kwargs)

            if not all_train_data:
                all_train_data = train_data
            elif train_data:
                all_train_data.examples = all_train_data.examples + train_data.examples
            if not all_validation_data:
                all_validation_data = validation_data
            elif validation_data:
                all_validation_data.examples = all_validation_data.examples + validation_data.examples
            if not all_test_data:
                all_test_data = test_data
            elif test_data:
                all_test_data.examples = all_test_data.examples + test_data.examples

        return Split(train=all_train_data, eval=all_validation_data, test=all_test_data), Split(
            train=train_path, eval=validation_path, test=test_path
        )


class OODDataset(CQA):
    name = 'ood'
    is_sequence_classification = True

    def __init__(self, path, lower=False, **kwargs):
        examples = []
        question = 'Is this sentence in-domain or out-domain?'

        dataset = load_dataset('csv', data_files=path, delimiter='\t', column_names=['tmp1', 'tmp2', 'sentence', 'label'])
        dataset = dataset['train']

        for data in dataset:
            context = data['sentence']
            answer = '1' if data['label'].strip() == '$ood ;' else '0'
            examples.append(Example.from_raw(make_example_id(self, len(examples)), context, question, answer, lower=lower))

        super().__init__(examples, **kwargs)

    @classmethod
    def splits(cls, root='.data', train='train', validation='eval', test='test', **kwargs):
        train_path = None if train is None else os.path.join(root, f'{train}.tsv')
        validation_path = None if validation is None else os.path.join(root, f'{validation}.tsv')
        test_path = None if test is None else os.path.join(root, f'{test}.tsv')

        train_data = None if train is None else cls(train_path, **kwargs)
        validation_data = None if validation is None else cls(validation_path, **kwargs)
        test_data = None if test is None else cls(test_path, **kwargs)

        return (
            Split(
                train=train_data,
                eval=validation_data,
                test=test_data,
            ),
            Split(train=train_path, eval=validation_path, test=test_path),
        )


class BiTODDataset(CQA):
    def __init__(self, path, *, make_example, **kwargs):
        subsample = kwargs.pop('subsample')
        examples = []

        with open(path) as fin:
            data = ujson.load(fin)['data']
            for turn in data:
                processed = make_example(turn, train_target=kwargs.get('train_target', False))
                if processed:
                    examples.append(processed)

                if subsample is not None and len(examples) >= subsample:
                    break

        super().__init__(examples, **kwargs)

        # do not sort eval/ test set so we can compute individual scores for each subtask (e2e_dialogue_score)
        self.eval_sort_key_fn = None

        # in e2e evaluation use 1 batch at a time
        if kwargs.get('e2e_evaluation', False):
            self.eval_batch_size_fn = default_batch_fn

    @classmethod
    def return_splits(cls, path='.data', train='train', validation='valid', test='test', **kwargs):
        train_path, validation_path, test_path = None, None, None
        if train:
            train_path = os.path.join(path, f'{train}.json')
        if validation:
            validation_path = os.path.join(path, f'{validation}.json')
        if test:
            test_path = os.path.join(path, 'test.json')

        train_data = None if train is None else cls(train_path, **kwargs)
        validation_data = None if validation is None else cls(validation_path, **kwargs)
        test_data = None if test is None else cls(test_path, **kwargs)

        return Split(train=train_data, eval=validation_data, test=test_data), Split(
            train=train_path, eval=validation_path, test=test_path
        )
