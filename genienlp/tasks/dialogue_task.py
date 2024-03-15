from ..data_utils.example import Example
from .base_task import BaseTask
from .dialogue_dataset import E2EDialogueDataset, E2EDialogueErrorClassificationDataset
from .registry import register_task


class E2EDialogueTask(BaseTask):
    def __init__(self, name, args):
        super().__init__(name, args)
        special_tokens_v1 = {
            '<user>',
            '<system>',
            '<API>',
            '<knowledge>',
            '<slot>',
            '<relation>',
            '<value>',
            '<sep>',
            '<unknow>',
            '<dialogue_state>',
        }
        special_tokens_v2 = {
            'USER:',
            'SYSTEM:',
            '<knowledge>',
            '<history>',
            '<state>',
            '#unknown',
            'DST:',
            'API:',
            'Response:',
        }
        special_tokens_v5 = {'AGENT_ACTS:'}
        special_tokens_v7 = {'ACTS:'}
        special_tokens_v9 = {'USER_ACTS:'}
        special_tokens_v11 = {'<endofknowledge>', '<endofhistory>', '<endofstate>'}
        special_tokens_v13 = {'AGENT_ACTS_PREV:'}
        special_tokens_v2_10 = {'<actions>', '<endofactions>', 'DA:', 'RG:'}
        self.special_tokens = (
            special_tokens_v1
            | special_tokens_v2
            | special_tokens_v5
            | special_tokens_v7
            | special_tokens_v9
            | special_tokens_v11
            | special_tokens_v13
            | special_tokens_v2_10
        )
        self._metrics = ['e2e_dialogue_score']

    def utterance_field(self):
        return 'context'

    def _make_example(self, turn, **kwargs):
        dial_id, turn_id, input_text, output_text, train_target = (
            turn['dial_id'],
            turn['turn_id'],
            turn['input_text'],
            turn['output_text'],
            turn['train_target'],
        )

        if kwargs.get('train_target', False) and train_target != kwargs['train_target']:
            return None

        example_id = '/'.join([dial_id, str(turn_id), train_target])

        return Example.from_raw(
            self.name + '/' + str(example_id), input_text, '', output_text, preprocess=self.preprocess_field, lower=False
        )

    def get_splits(self, root, **kwargs):
        kwargs['e2e_evaluation'] = self.args.e2e_dialogue_evaluation
        return E2EDialogueDataset.return_splits(path=root, make_example=self._make_example, **kwargs)


@register_task('risawoz')
class RiSAWOZ(E2EDialogueTask):
    def __init__(self, name, args):
        super().__init__(name, args)
        self.dataset_name = 'Risawoz'


@register_task('bitod')
class BiTOD(E2EDialogueTask):
    def __init__(self, name, args):
        super().__init__(name, args)
        self.dataset_name = 'Bitod'


@register_task('bitod_nlg')
class BiTODNLG(BiTOD):
    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['casedbleu']

    def get_splits(self, root, **kwargs):
        kwargs['train_target'] = 'rg'
        kwargs['e2e_evaluation'] = self.args.e2e_dialogue_evaluation
        return E2EDialogueDataset.return_splits(path=root, make_example=self._make_example, **kwargs)


@register_task('bitod_dst')
class BiTODDST(BiTOD):
    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['dst_em', 'jga']

    def get_splits(self, root, **kwargs):
        kwargs['train_target'] = 'dst'
        kwargs['e2e_evaluation'] = self.args.e2e_dialogue_evaluation
        return E2EDialogueDataset.return_splits(path=root, make_example=self._make_example, **kwargs)
