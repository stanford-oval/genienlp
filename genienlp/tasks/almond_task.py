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
import logging
import os

from ..data_utils.almond_utils import detokenize_cjk_chars, is_device, is_entity_marker, tokenize_cjk_chars
from ..data_utils.example import Example
from .almond_dataset import AlmondDataset
from .base_task import BaseTask
from .registry import register_task

logger = logging.getLogger(__name__)


class BaseAlmondTask(BaseTask):
    """Base class for the Almond semantic parsing task
    i.e. natural language to formal language (ThingTalk) mapping"""

    def __init__(self, name, args):
        super().__init__(name, args)
        self.args = args
        self._metrics = ['em', 'sm', 'f1']

        self.need_attention_scores = False

        self._almond_detokenize_sentence = args.almond_detokenize_sentence

    @property
    def utterance_field(self):
        return NotImplementedError()

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, new_metrics):
        self._metrics = new_metrics

    def _is_program_field(self, field_name):
        raise NotImplementedError()

    def _make_example(self, parts, dir_name, **kwargs):
        raise NotImplementedError()

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, **kwargs)

    def batch_postprocess_prediction_ids(self, batch_example_ids, batch_src_ids, batch_tgt_ids, **kwargs):
        return batch_tgt_ids, None

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

    def preprocess_field(self, sentence, field_name=None, answer=None, example_id=None, preprocess_entities=True):
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
            if  is_program and (is_device(token) or is_entity_marker(token)):
                if token.startswith('QUOTED_STRING_'):
                    token = token[len('QUOTED_') :]
                elif token.startswith('GENERIC_ENTITY_'):
                    token = token[len('GENERIC_') :]

                self.special_tokens.add(token)
            new_tokens.append(token)
        tokens = new_tokens
        new_sentence = ' '.join(tokens)

        if self._almond_detokenize_sentence:

            # BERT tokenizers by default add whitespace around any CJK character
            # SPM-based tokenizers are trained on raw text and do better when receive untokenized text
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

                if not is_entity_marker(token) and not is_device(token):
                    for word in token.split('_'):
                        new_tokens.append(word)
                else:
                    new_tokens.append(token)
            new_sentence = ' '.join(new_tokens)

        new_sentence = new_sentence.strip()

        return new_sentence




@register_task('almond_natural_seq2seq')
class NaturalSeq2Seq(BaseAlmondTask):
    """
    The Almond sequence to sequence task where both sequences are natural language.
    In this task entities (see ENTITY_REGEX) are not preprocessed in contrast to paraphrasing and translation tasks.
    Paraphrasing and translation inherit from this class.
    """

    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['bleu', 'em']

    def _is_program_field(self, field_name):
        return False

    @property
    def utterance_field(self):
        return 'context'

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
        return Example.from_raw(
            self.name + '/' + example_id, context, question, answer, preprocess=self.preprocess_field, lower=False
        )

    def preprocess_field(self, sentence, field_name=None, answer=None, example_id=None, preprocess_entities=False):
        return super().preprocess_field(sentence, field_name, answer, example_id, preprocess_entities=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, **kwargs)



@register_task('reverse_almond')
class ReverseAlmond(BaseAlmondTask):
    """Reverse Almond semantic parsing task
    i.e. formal language to natural language mapping"""

    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['bleu', 'em']

    @property
    def utterance_field(self):
        return 'context'

    def _is_program_field(self, field_name):
        return field_name == 'context'

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        example_id, sentence, target_code = parts
        question = 'translate from thingtalk to english'
        context = target_code
        answer = sentence
        return Example.from_raw(
            self.name + '/' + example_id, context, question, answer, preprocess=self.preprocess_field, lower=False
        )


# TODO add a similar preprocessing step to Multilingual dialogue tasks as well
class BaseAlmondDialogueNLUTask(BaseAlmondTask):
    def preprocess_field(self, sentence, field_name=None, answer=None, example_id=None, preprocess_entities=True):
        if not sentence:
            return sentence

        # remove the $dialogue at the start of the dialogue
        # this is safe because we know we're processing dialogues, so the answer
        # always starts with $dialogue and the context is either `null` or also
        # starts with $dialogue
        if field_name in ['context', 'answer'] and sentence.startswith('$dialogue '):
            sentence = sentence[len('$dialogue ') :]
        return super().preprocess_field(sentence, field_name, answer, example_id, preprocess_entities)

    def postprocess_prediction(self, example_id, prediction):
        prediction = super().postprocess_prediction(example_id, prediction)
        if not prediction.startswith('$'):
            return '$dialogue ' + prediction
        return prediction





@register_task('almond_dialogue_nlg')
class AlmondDialogueNLG(BaseAlmondTask):
    """Multi-turn NLG task for Almond dialogues
    (generate the system utterance, given the current state of the conversation
    and the desired system dialogue act)
    """

    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['bleu']

    def _is_program_field(self, field_name):
        return field_name in ('context', 'question')

    @property
    def utterance_field(self):
        return 'answer'

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task
        example_id, context, sentence, target_code = parts
        question = target_code
        answer = sentence
        return Example.from_raw(
            self.name + '/' + example_id, context, question, answer, preprocess=self.preprocess_field, lower=False
        )

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/nlg'), make_example=self._make_example, **kwargs)


@register_task('almond_dialogue_policy')
class AlmondDialoguePolicy(BaseAlmondTask):
    """Multi-turn dialogue policy task for Almond dialogues
    (generate the next dialogue act, given the current state of the conversation)
    """

    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['em', 'f1']

    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    @property
    def utterance_field(self):
        return 'question'

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task, and the sentence is intentionally ignored
        example_id, context, _sentence, target_code = parts
        question = 'what should the agent do ?'
        answer = target_code
        return Example.from_raw(
            self.name + '/' + example_id, context, question, answer, preprocess=self.preprocess_field, lower=False
        )

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)
