#
# Copyright (c) 2019, The Board of Trustees of the Leland Stanford Junior University
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

import os
import torch
import logging
from collections import defaultdict

from .base_task import BaseTask
from .registry import register_task
from .generic_dataset import CQA, default_batch_fn, input_then_output_len, input_tokens_fn
from ..data_utils.example import Example
from ..data_utils.progbar import progress_bar
from .almond_utils import ISO_to_LANG, is_device, is_entity, is_entity_marker, process_id, tokenize_cjk_chars, detokenize_cjk_chars
from ..paraphrase.data_utils import input_heuristics, output_heuristics

from .base_dataset import Split

logger = logging.getLogger(__name__)


class AlmondDataset(CQA):
    """Obtaining dataset for Almond semantic parsing task"""

    base_url = None

    def __init__(self, path, *, make_example, subsample=None, cached_path=None, skip_cache=False, cache_input_data=False, **kwargs):
        
        #TODO fix cache_path for multilingual task
        cache_name = os.path.join(cached_path, os.path.basename(path), str(subsample))
        dir_name = os.path.basename(os.path.dirname(path))

        if os.path.exists(cache_name) and not skip_cache:
            logger.info(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            examples = []
            n = 0
            with open(path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    n += 1

            max_examples = min(n, subsample) if subsample is not None else n
            for line in progress_bar(open(path, 'r', encoding='utf-8'), total=max_examples, desc="Reading Dataset"):
                parts = line.strip().split('\t')
                examples.append(make_example(parts, dir_name, **kwargs))
                if len(examples) >= max_examples:
                    break
            os.makedirs(os.path.dirname(cache_name), exist_ok=True)
            if cache_input_data:
                logger.info(f'Caching data to {cache_name}')
                torch.save(examples, cache_name)

        super().__init__(examples, **kwargs)
        

    @classmethod
    def return_splits(cls, path, train='train', validation='eval', test='test', **kwargs):

        """Create dataset objects for splits of the ThingTalk dataset.
        Arguments:
            path: path to directory where data splits reside
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'eval'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        
        train_data = None if train is None else cls(os.path.join(path, train + '.tsv'), **kwargs)
        validation_data = None if validation is None else cls(os.path.join(path, validation + '.tsv'), **kwargs)
        test_data = None if test is None else cls(os.path.join(path, test + '.tsv'), **kwargs)

        aux_data = None
        do_curriculum = kwargs.get('curriculum', False)
        if do_curriculum:
            kwargs.pop('curriculum')
            aux_data = cls(os.path.join(path, 'aux' + '.tsv'), **kwargs)
        
        return Split(train=None if train is None else train_data,
                     eval=None if validation is None else validation_data,
                     test=None if test is None else test_data,
                     aux=None if do_curriculum is None else aux_data)
    


class BaseAlmondTask(BaseTask):
    """Base class for the Almond semantic parsing task
        i.e. natural language to formal language (ThingTalk) mapping"""

    def __init__(self, name, args):
        super().__init__(name, args)
        self._almond_has_multiple_programs = args.almond_has_multiple_programs
        self._almond_detokenize_sentence = args.almond_detokenize_sentence

    @property
    def metrics(self):
        return ['em', 'sm', 'f1']

    def _is_program_field(self, field_name):
        raise NotImplementedError()

    def _make_example(self, parts, dir_name, **kwargs):
        raise NotImplementedError()

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, **kwargs)

    def postprocess_prediction(self, example_id, prediction):
    
        if self._almond_detokenize_sentence:
            # To make genienlp transparent to the tokenization done by genie-toolkit
            # We tokenize prediction here by adding whitespace between each CJK character
            prediction = tokenize_cjk_chars(prediction)
        
        new_tokens = []
        for token in prediction.split():
            if token.startswith('STRING_'):
                token = 'QUOTED_' + token
            elif token.startswith('ENTITY_'):
                token = 'GENERIC_' + token
            new_tokens.append(token)
        new_prediction = ' '.join(new_tokens)
        return new_prediction

    def preprocess_field(self, sentence, field_name=None):
        if self.override_context is not None and field_name == 'context':
            return self.override_context
        if self.override_question is not None and field_name == 'question':
            return self.override_question
        if not sentence:
            return ''

        tokens = sentence.split(' ')
        is_program = self._is_program_field(field_name)
        new_tokens = []
        for token in tokens:
            if is_entity(token) or (is_program and (is_device(token) or is_entity_marker(token))):
                if token.startswith('QUOTED_STRING_'):
                    token = token[len('QUOTED_'):]
                elif token.startswith('GENERIC_ENTITY_'):
                    token = token[len('GENERIC_'):]

                self.special_tokens.add(token)
            new_tokens.append(token)
        tokens = new_tokens
        new_sentence = ' '.join(tokens)

        if self._almond_detokenize_sentence:
            
            # BERT tokenizers by default add whitespace around any CJK character
            # SPM-based tokenizers are trained on raw text and do better when recieve untokenized text
            # In genienlp we detokenize CJK characters and leave tokenization to the model's tokenizer
            # NOTE: input datasets for almond are usually pretokenized using genie-toolkit which
            # inserts whitespace around any CJK character. This detokenization ensures that SPM-based tokenizers
            # see the text without space between those characters
            new_sentence = detokenize_cjk_chars(new_sentence)
            tokens = new_sentence.split(' ')
            
            new_sentence = ''
            in_string = False
            for token in tokens:
                if is_program:
                    if token == '"':
                        in_string = not in_string
                    if not in_string:
                        new_sentence += ' ' + token
                        continue
                if token in (',', '.', '?', '!', ':', ')', ']', '}') or token.startswith("'"):
                    new_sentence += token
                else:
                    new_sentence += ' ' + token
        elif is_program and field_name != 'answer':
            new_tokens = []
            in_string = False
            for token in tokens:
                if token == '"':
                    in_string = not in_string
                if in_string:
                    new_tokens.append(token)
                    continue

                if not is_entity(token) and not is_entity_marker(token) and \
                        not is_device(token):
                    for word in token.split('_'):
                        new_tokens.append(word)
                else:
                    new_tokens.append(token)
            new_sentence = ' '.join(new_tokens)

        return new_sentence


@register_task('almond')
class Almond(BaseAlmondTask):
    """The Almond semantic parsing task
    i.e. natural language to formal language (ThingTalk) mapping"""

    def _is_program_field(self, field_name):
        return field_name == 'answer'

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        if self._almond_has_multiple_programs:
            example_id, sentence, target_code = parts[:3]
        else:
            example_id, sentence, target_code = parts
        question = 'translate from english to thingtalk'
        context = sentence
        answer = target_code
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

@register_task('natural_seq2seq')
class NaturalSeq2Seq(BaseAlmondTask):
    """The Almond seqeunce to sequence task where both sequences are natural language
    i.e. no ThingTalk program. Paraphrasing and translation are examples of this task"""

    @property
    def metrics(self):
        return ['bleu', 'em', 'nf1']

    def _is_program_field(self, field_name):
        return False

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant
        if len(parts) == 2:
            input_sequence, target_sequence = parts
            example_id = "id-null"
        elif len(parts) == 3:
            example_id, input_sequence, target_sequence = parts
        else:
            raise ValueError(f'Input file contains line with {len(parts)} parts: {str(parts)}')
        question = 'translate from input to output'
        context = input_sequence
        answer = target_sequence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, **kwargs)


@register_task('paraphrase')
class Paraphrase(NaturalSeq2Seq):
    """The Almond paraphrasing task. Applies the necessary preprocessing for special tokens and case changes.
    Should only be used at prediction time.
    """

    def __init__(self, name, args):
        super().__init__(name, args)
        self.reverse_maps = {}

    @property
    def metrics(self):
        return ['bleu']

    def postprocess_prediction(self, example_id, prediction):
        return output_heuristics(prediction, self.reverse_maps[example_id])


    def _make_example(self, parts, dir_name=None, **kwargs):
        if len(parts) == 3:
            example_id, sentence, thingtalk = parts
        elif len(parts) == 4:
            example_id, _, sentence, thingtalk = parts # ignore dialogue context
        else:
            raise ValueError(f'Input file contains line with {len(parts)} parts: {str(parts)}')

        example_id = self.name + '/' + example_id

        sentence, reverse_map = input_heuristics(sentence, thingtalk=thingtalk, is_cased=True)
        # this task especially needs example ids to be unique
        if example_id in self.reverse_maps:
            example_id += '.'
        self.reverse_maps[example_id] = reverse_map

        question = 'translate from input to output'
        context = sentence
        answer = sentence # means we calculate self-bleu
        
        return Example.from_raw(example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, **kwargs)


@register_task('contextual_almond')
class ContextualAlmond(BaseAlmondTask):
    """Contextual Almond semantic parsing task
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts[:4]
        else:
            example_id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)


@register_task('reverse_almond')
class ReverseAlmond(BaseAlmondTask):
    """Reverse Almond semantic parsing task
    i.e. formal language to natural language mapping"""

    @property
    def metrics(self):
        return ['bleu', 'em']

    def _is_program_field(self, field_name):
        return field_name == 'context'

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        example_id, sentence, target_code = parts
        question = 'translate from thingtalk to english'
        context = target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

# TODO add a similar preprocessing step to Multilingual dialogue tasks as well
class BaseAlmondDialogueNLUTask(BaseAlmondTask):
    def preprocess_field(self, sentence, field_name=None):
        if not sentence:
            return sentence

        # remove the $dialogue at the start of the dialogue
        # this is safe because we know we're processing dialogues, so the answer
        # always starts with $dialogue and the context is either `null` or also
        # starts with $dialogue
        if field_name == 'context' and sentence.startswith('$dialogue '):
            sentence = sentence[len('$dialogue '):]
        if field_name == 'answer':
            if sentence.startswith('$dialogue '):
                sentence = sentence[len('$dialogue '):]

        return super().preprocess_field(sentence, field_name)

    def postprocess_prediction(self, example_id, prediction):
        prediction = super().postprocess_prediction(example_id, prediction)
        if not prediction.startswith('$'):
            return '$dialogue ' + prediction
        return prediction

@register_task('almond_dialogue_nlu')
class AlmondDialogueNLU(BaseAlmondDialogueNLUTask):
    """Multi-turn NLU task for Almond dialogues
    (translate the user utterance to a formal representation, given the current
    state of the conversation)
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts[:4]
        else:
            example_id, context, sentence, target_code = parts

        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/user'), make_example=self._make_example, **kwargs)


@register_task('almond_dialogue_nlu_agent')
class AlmondDialogueNLUAgent(BaseAlmondDialogueNLUTask):
    """Multi-turn NLU task for Almond dialogues, for the agent utterance
    (translate the agent utterance to a formal representation, given the current
    state of the conversation).
    This is used to facilitate annotation of human-human dialogues.
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts[:4]
        else:
            example_id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)


@register_task('almond_dialogue_nlg')
class AlmondDialogueNLG(BaseAlmondTask):
    """Multi-turn NLG task for Almond dialogues
    (generate the system utterance, given the current state of the conversation
    and the desired system dialogue act)
    """
    def _is_program_field(self, field_name):
        return field_name == 'context'

    @property
    def metrics(self):
        return ['bleu']

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task
        example_id, context, sentence, target_code = parts
        question = 'what should the agent say ?'
        context = context + ' ' + target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)


@register_task('almond_dialogue_policy')
class AlmondDialoguePolicy(BaseAlmondTask):
    """Multi-turn dialogue policy task for Almond dialogues
    (generate the next dialogue act, given the current state of the conversation)
    """
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    @property
    def metrics(self):
        return ['em', 'f1']

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task, and the sentence is intentionally ignored
        example_id, context, _sentence, target_code = parts
        question = 'what should the agent do ?'
        context = context
        answer = target_code
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)
    

class BaseAlmondMultiLingualTask(BaseAlmondTask):
    """ Base task for MultiLingual Almond
    """
    def get_train_processed_ids(self, split):
        all_ids = []
        for ex in split.examples:
            all_ids.append(process_id(ex))
        return all_ids

    def combine_datasets(self, datasets, sort_key_fn, batch_size_fn, used_fields, groups):
        splits = defaultdict()
    
        for field in used_fields:
            all_examples = []
            for dataset in datasets:
                all_examples.extend(getattr(dataset, field).examples)
        
            splits[field] = CQA(all_examples, sort_key_fn=sort_key_fn, batch_size_fn=batch_size_fn, groups=groups)
    
        return Split(train=splits.get('train'),
                     eval=splits.get('eval'),
                     test=splits.get('test'),
                     aux=splits.get('aux'))

    def get_splits(self, root, **kwargs):
        all_datasets = []
        # number of directories to read data from
        all_dirs = kwargs['all_dirs'].split('+')
        
        for dir in all_dirs:
            almond_dataset = AlmondDataset.return_splits(path=os.path.join(root, 'almond/multilingual/{}'.format(dir)),
                                                         make_example=self._make_example, **kwargs)
            all_datasets.append(almond_dataset)
            
        used_fields = [field for field in all_datasets[0]._fields if getattr(all_datasets[0], field) is not None]
        
        assert len(all_datasets) >= 1
        if kwargs.get('sentence_batching'):
            for field in used_fields:
                lengths = list(map(lambda dataset: len(getattr(dataset, field)), all_datasets))
                assert len(set(lengths)) == 1, 'When using sentence batching your datasets should have the same size.'
            if 'train' in used_fields:
                ids_sets = list(map(lambda dataset: set(self.get_train_processed_ids(dataset.train)), all_datasets))
                id_set_base = set(ids_sets[0])
                for id_set in ids_sets:
                    assert set(id_set) == id_set_base, 'When using sentence batching your datasets should have matching ids'
            
            sort_key_fn = process_id
            batch_size_fn = default_batch_fn
        else:
            # use default values for `sort_key_fn` and `batch_size_fn`
            sort_key_fn = input_then_output_len
            batch_size_fn = input_tokens_fn
            
        groups = len(all_datasets) if kwargs.get('sentence_batching') else None
        
        if kwargs.get('separate_eval') and (all_datasets[0].eval or all_datasets[0].test):
            return all_datasets
        else:
            return self.combine_datasets(all_datasets, sort_key_fn, batch_size_fn, used_fields, groups)


@register_task('almond_multilingual')
class AlmondMultiLingual(BaseAlmondMultiLingualTask):
    def __init__(self, name, args):
        super().__init__(name, args)
        self.lang_as_question = args.almond_lang_as_question

    def _is_program_field(self, field_name):
        return field_name == 'answer'
    
    def _make_example(self, parts, dir_name, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, sentence, target_code = parts[:3]
        else:
            example_id, sentence, target_code = parts
        language = ISO_to_LANG.get(dir_name, 'English').lower()
        if self.lang_as_question:
            question = 'translate from {} to thingtalk'.format(language)
        else:
            question = 'translate from english to thingtalk'
        context = sentence
        answer = target_code
        return Example.from_raw(self.name + '/' + dir_name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)


@register_task('almond_dialogue_multilingual_nlu')
class AlmondDialogMultiLingualNLU(BaseAlmondMultiLingualTask):
    """Multi-turn NLU task (user and agent) for MultiLingual Almond dialogues
    """

    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts
        else:
            example_id, context, sentence, target_code = parts[:4]
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + dir_name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)


@register_task('almond_dialogue_multilingual_nlg')
class AlmondDialogMultiLingualNLG(BaseAlmondTask):
    """Multi-turn NLG task (agent) for MultiLingual Almond dialogues
    """
    def _is_program_field(self, field_name):
        return field_name == 'context'

    @property
    def metrics(self):
        return ['bleu']

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task
        example_id, context, sentence, target_code = parts
        question = 'what should the agent say ?'
        context = context + ' ' + target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + dir_name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

