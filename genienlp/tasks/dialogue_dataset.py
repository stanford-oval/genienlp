import os

import json

from .base_dataset import Split
from .generic_dataset import CQA


class E2EDialogueDataset(CQA):
    def __init__(self, path, *, make_example, **kwargs):
        subsample = kwargs.pop('subsample')
        examples = []

        with open(path) as fin:
            data = json.load(fin)['data']
            for turn in data:
                processed = make_example(turn, train_target=kwargs.get('train_target', False))
                if processed:
                    examples.append(processed)

                if subsample is not None and len(examples) >= subsample:
                    break

        super().__init__(examples, **kwargs)

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
