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
from collections import defaultdict

import torch

from genienlp.data_utils.almond_utils import split_text_into_sentences

from ..data_utils.almond_utils import (
    ISO_to_LANG,
    detokenize_cjk_chars,
    is_device,
    is_entity,
    is_entity_marker,
    process_id,
    tokenize_cjk_chars,
)
from ..data_utils.example import Example
from ..model_utils.translation import align_and_replace, compute_attention
from ..paraphrase.data_utils import input_heuristics, output_heuristics
from .almond_dataset import AlmondDataset
from .base_dataset import Split
from .base_task import BaseTask
from .generic_dataset import CQA, all_tokens_fn, default_batch_fn, input_then_output_len
from .registry import register_task

logger = logging.getLogger(__name__)


class BaseAlmondTask(BaseTask):
    """Base class for the Almond semantic parsing task
    i.e. natural language to formal language (ThingTalk) mapping"""

    def __init__(self, name, args):
        super().__init__(name, args)
        self.args = args
        self._metrics = ['em', 'sm', 'f1']
        self.no_feature_fields = ['answer']
        if self.utterance_field == 'question':
            self.no_feature_fields.append('context')
        else:
            self.no_feature_fields.append('question')

        self.need_attention_scores = False

        self._almond_has_multiple_programs = args.almond_has_multiple_programs
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
            if (is_entity(token) and preprocess_entities) or (is_program and (is_device(token) or is_entity_marker(token))):
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

                if not is_entity(token) and not is_entity_marker(token) and not is_device(token):
                    for word in token.split('_'):
                        new_tokens.append(word)
                else:
                    new_tokens.append(token)
            new_sentence = ' '.join(new_tokens)

        new_sentence = new_sentence.strip()

        return new_sentence


@register_task('almond')
class Almond(BaseAlmondTask):
    """The Almond semantic parsing task
    i.e. natural language to formal language (ThingTalk) mapping"""

    def _is_program_field(self, field_name):
        return field_name == 'answer'

    @property
    def utterance_field(self):
        return 'context'

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
        return Example.from_raw(
            self.name + '/' + example_id, context, question, answer, preprocess=self.preprocess_field, lower=False
        )


@register_task('almond_natural_seq2seq')
class NaturalSeq2Seq(BaseAlmondTask):
    """
    The Almond sequence to sequence task where both sequences are natural language.
    In this task entities (see ENTITY_REGEX) are not preprocessed in contrast to paraphrasing and translation tasks.
    Paraphrasing and translation inherit from this class.
    """

    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['bleu', 'em', 'nf1']

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


@register_task('almond_paraphrase')
class Paraphrase(NaturalSeq2Seq):
    """The Almond paraphrasing task. Applies the necessary preprocessing for special tokens and case changes.
    Can be used at prediction and training time. Training is still experimental.
    """

    def __init__(self, name, args):
        super().__init__(name, args)
        self.reverse_maps = {}
        self._metrics = ['bleu']

    def postprocess_prediction(self, example_id, prediction):
        return output_heuristics(prediction, self.reverse_maps[example_id])

    def _make_example(self, parts, dir_name=None, **kwargs):
        if len(parts) == 3:
            example_id, sentence, thingtalk = parts
        elif len(parts) == 4:
            example_id, _, sentence, thingtalk = parts  # ignore dialogue context
        else:
            raise ValueError(f'Input file contains line with {len(parts)} parts: {str(parts)}')

        example_id = self.name + '/' + example_id

        sentence, reverse_map = input_heuristics(sentence, thingtalk=thingtalk, is_cased=True)
        # this task especially needs example ids to be unique
        while example_id in self.reverse_maps:
            example_id += '.'
        self.reverse_maps[example_id] = reverse_map

        question = 'translate from input to output'
        context = sentence
        answer = sentence  # means we calculate self-bleu

        return Example.from_raw(example_id, context, question, answer, preprocess=self.preprocess_field, lower=False)


@register_task('almond_translate')
class Translate(NaturalSeq2Seq):
    """
    Almond translation task: Translate a sentence from one language to another.
    Can be used at prediction and training time. Training is still experimental.
    """

    def __init__(self, name, args):
        super().__init__(name, args)
        self.input_spans = {}
        self.all_ids = set()
        self._metrics = ['casedbleu']

        # only requires cross_attention scores for alignment
        self.need_attention_scores = bool(self.args.do_alignment)

    def construct_id2span_mapping(self, example_id, sentence, field_name):
        assert field_name in ['context', 'question']
        # translation task constructs a dictionary mapping ids to entity spans in the sentence
        # this ensures the ids are unique
        while field_name + '-' + example_id in self.all_ids:
            example_id += '.'

        self.all_ids.add(field_name + '-' + example_id)

        src_quotation_symbol = '"'
        src_tokens = sentence.split(" ")
        src_spans_ind = [index for index, token in enumerate(src_tokens) if token == src_quotation_symbol]

        if len(src_spans_ind) % 2 != 0:
            raise ValueError(f'Corrupted span in sentence: [{sentence}]')

        if self.args.align_preserve_input_quotation:
            src_spans = [(src_spans_ind[i] + 1, src_spans_ind[i + 1] - 1) for i in range(0, len(src_spans_ind), 2)]
        else:
            src_tokens = [token for token in src_tokens if token != src_quotation_symbol]
            src_spans = [
                (src_spans_ind[i] + 1 - (i + 1), src_spans_ind[i + 1] - 1 - (i + 1)) for i in range(0, len(src_spans_ind), 2)
            ]

        # remove illegal src_spans (caused by inputs such as " ")
        src_spans = [span for span in src_spans if span[0] <= span[1]]

        sentence = " ".join(src_tokens)
        src_spans_flatten = [val for tup in src_spans for val in tup]

        # append question spans to context spans
        if example_id in self.input_spans:
            self.input_spans[example_id] += src_spans_flatten
        else:
            self.input_spans[example_id] = src_spans_flatten

        return example_id, sentence

    def preprocess_field(self, sentence, field_name=None, answer=None, example_id=None, preprocess_entities=True):
        return super().preprocess_field(sentence, field_name, answer, preprocess_entities)

    def _make_example(self, parts, dir_name=None, **kwargs):
        # answer has to be provided by default unless doing prediction
        no_answer = getattr(self.args, 'translate_no_answer', False)
        split_sentence = getattr(self.args, 'translate_example_split', False)
        src_lang = kwargs.get('src_lang', 'en')

        example_id = 'id-null'
        question = 'translate from input to output'

        if no_answer:
            if len(parts) == 1:
                context = parts
            elif len(parts) == 2:
                example_id, context = parts
            elif len(parts) == 3:
                example_id, context, question = parts
            elif len(parts) == 4:
                raise ValueError(f'Input file contains a line with {len(parts)} parts: {str(parts)}')
        else:
            if len(parts) == 2:
                context, answer = parts
            elif len(parts) == 3:
                example_id, context, answer = parts
            elif len(parts) == 4:
                example_id, context, question, answer = parts
            else:
                raise ValueError(f'Input file contains a line with {len(parts)} parts: {str(parts)}')

        # no answer is provided
        if no_answer:
            answer = '.'

        contexts = []
        src_char_spans = None
        if split_sentence:
            if self.args.do_alignment:
                src_quotation_symbol = '"'
                src_char_spans_ind = [index for index, char in enumerate(context) if char == src_quotation_symbol]
                src_char_spans = [
                    (src_char_spans_ind[i], src_char_spans_ind[i + 1]) for i in range(0, len(src_char_spans_ind), 2)
                ]
            contexts = split_text_into_sentences(context, src_lang, src_char_spans)

        if len(contexts) > 1:
            examples = []
            for i, text in enumerate(contexts):
                ex_id, text = self.construct_id2span_mapping(self.name + '/' + example_id + f'@{i}', text, 'context')
                examples.append(
                    Example.from_raw(
                        ex_id,
                        text,
                        question,
                        answer,
                        preprocess=self.preprocess_field,
                        lower=False,
                    )
                )
        else:
            ex_id, context = self.construct_id2span_mapping(self.name + '/' + example_id, context, 'context')
            examples = Example.from_raw(ex_id, context, question, answer, preprocess=self.preprocess_field, lower=False)

        return examples

    def batch_postprocess_prediction_ids(self, batch_example_ids, batch_src_ids, batch_tgt_ids, **kwargs):
        numericalizer = kwargs.pop('numericalizer')
        cross_attentions = kwargs.pop('cross_attentions')
        tgt_lang = kwargs.pop('tgt_lang')
        date_parser = kwargs.pop('date_parser')
        num_outputs = len(batch_tgt_ids) // len(batch_src_ids)

        # TODO _tokenizer should not be private
        tokenizer = numericalizer._tokenizer

        all_src_tokens = numericalizer.convert_ids_to_tokens(batch_src_ids, skip_special_tokens=False)
        all_tgt_tokens = numericalizer.convert_ids_to_tokens(batch_tgt_ids, skip_special_tokens=False)

        # remove input_prefix from the beginning of src_tokens and shift layer_attention
        len_prefix_wp = len(tokenizer.tokenize(numericalizer.input_prefix))
        all_src_tokens = [tokens[len_prefix_wp:] for tokens in all_src_tokens]
        cross_attentions = cross_attentions[:, :, :, len_prefix_wp:]

        cross_attention_pooled = compute_attention(cross_attentions, att_pooling=self.args.att_pooling, dim=1)

        all_text_outputs = []
        # post-process predictions ids
        for i, (tgt_tokens, cross_att) in enumerate(zip(all_tgt_tokens, cross_attention_pooled)):

            src_tokens = all_src_tokens[i // num_outputs]
            example_id = batch_example_ids[i // num_outputs]

            # shift target tokens left to match the attention positions (since eos_token is prepended not generated)
            if tgt_tokens[0] in tokenizer.all_special_tokens:
                tgt_tokens = tgt_tokens[1:]

            # remove all beginning special tokens from target and shift attention too
            while tgt_tokens[0] in tokenizer.all_special_tokens:
                tgt_tokens = tgt_tokens[1:]
                cross_att = cross_att[1:, :]

            # remove all beginning special tokens from source and shift attention too
            while src_tokens[0] in tokenizer.all_special_tokens:
                src_tokens = src_tokens[1:]
                cross_att = cross_att[:, 1:]

            # remove all trailing special tokens from source
            while src_tokens[-1] in tokenizer.all_special_tokens:
                src_tokens = src_tokens[:-1]

            # remove all trailing special tokens from target
            while tgt_tokens[-1] in tokenizer.all_special_tokens:
                tgt_tokens = tgt_tokens[:-1]

            # crop to match src and tgt new lengths
            cross_att = cross_att[: len(tgt_tokens), : len(src_tokens)]

            # plot cross-attention heatmap
            if getattr(self.args, 'plot_heatmaps', False):
                import matplotlib.pyplot as plt
                import seaborn as sns

                graph = sns.heatmap(torch.log(cross_att), xticklabels=src_tokens, yticklabels=tgt_tokens)
                graph.set_xticklabels(graph.get_xmajorticklabels(), fontsize=12)
                graph.set_yticklabels(graph.get_ymajorticklabels(), fontsize=12)

                plt.savefig(
                    os.path.join(
                        getattr(self.args, 'save', self.args.eval_dir),
                        f'heatmap_{batch_example_ids[i].replace("/", "-")}',
                    )
                )
                plt.show()

            if self.args.do_alignment:
                src_spans = self.input_spans[example_id]
                try:
                    text = align_and_replace(
                        src_tokens,
                        tgt_tokens,
                        cross_att,
                        src_spans,
                        tgt_lang,
                        tokenizer,
                        self.args.align_remove_output_quotation,
                        date_parser=date_parser,
                    )
                except Exception as e:
                    logger.error(str(e))
                    logger.error(f'Alignment failed for src_tokens: [{src_tokens}] and tgt_tokens: [{tgt_tokens}]')
                    text = tokenizer.convert_tokens_to_string(tgt_tokens)

            else:
                text = tokenizer.convert_tokens_to_string(tgt_tokens)

            all_text_outputs.append(text)

        with tokenizer.as_target_tokenizer():
            partial_batch_prediction_ids = tokenizer.batch_encode_plus(all_text_outputs, padding=True, return_tensors='pt')[
                'input_ids'
            ]

        return partial_batch_prediction_ids, all_text_outputs


@register_task('contextual_almond')
class ContextualAlmond(BaseAlmondTask):
    """Contextual Almond semantic parsing task"""

    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    @property
    def utterance_field(self):
        return 'question'

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts[:4]
        else:
            example_id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(
            self.name + '/' + example_id, context, question, answer, preprocess=self.preprocess_field, lower=False
        )


@register_task('reverse_almond')
class ReverseAlmond(BaseAlmondTask):
    """Reverse Almond semantic parsing task
    i.e. formal language to natural language mapping"""

    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['blue', 'em']

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


@register_task('almond_dialogue_nlu')
class AlmondDialogueNLU(BaseAlmondDialogueNLUTask):
    """Multi-turn NLU task for Almond dialogues
    (translate the user utterance to a formal representation, given the current
    state of the conversation)
    """

    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    @property
    def utterance_field(self):
        return 'question'

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts[:4]
        else:
            example_id, context, sentence, target_code = parts

        answer = target_code
        question = sentence
        return Example.from_raw(
            self.name + '/' + example_id, context, question, answer, preprocess=self.preprocess_field, lower=False
        )

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

    @property
    def utterance_field(self):
        return 'question'

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts[:4]
        else:
            example_id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(
            self.name + '/' + example_id, context, question, answer, preprocess=self.preprocess_field, lower=False
        )

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example, **kwargs)


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


class BaseAlmondMultiLingualTask(BaseAlmondTask):
    """Base task for MultiLingual Almond"""

    def get_train_processed_ids(self, split):
        all_ids = []
        for ex in split.examples:
            all_ids.append(process_id(ex))
        return all_ids

    def combine_datasets(self, datasets, all_paths, sort_key_fn, batch_size_fn, used_fields, groups):
        splits = defaultdict()
        # paths = defaultdict()

        for field in used_fields:
            # choose one path and replace dir name with 'combined'
            # paths[field] = '/'.join([getattr(all_paths[0], field).rsplit('/', 2)[0], 'combined', getattr(all_paths[0], field).rsplit('/', 2)[2]])

            all_examples = []
            for dataset in datasets:
                all_examples.extend(getattr(dataset, field).examples)

            splits[field] = CQA(all_examples, sort_key_fn=sort_key_fn, batch_size_fn=batch_size_fn, groups=groups)

        return Split(train=splits.get('train'), eval=splits.get('eval'), test=splits.get('test'), aux=splits.get('aux'))
        # Split(train=paths.get('train'), eval=paths.get('eval'), test=paths.get('test'), aux=paths.get('aux'))

    def get_splits(self, root, **kwargs):
        all_datasets = []
        all_paths = []
        # number of directories to read data from
        all_dirs = kwargs['all_dirs'].split('+')

        for dir in all_dirs:
            splits, paths = AlmondDataset.return_splits(
                path=os.path.join(root, 'almond/{}'.format(dir)), make_example=self._make_example, **kwargs
            )
            all_datasets.append(splits)
            all_paths.append(paths)

        used_fields = [field for field in all_datasets[0]._fields if getattr(all_datasets[0], field) is not None]

        assert len(all_datasets) >= 1
        if getattr(self.args, 'sentence_batching', False):
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
            batch_size_fn = all_tokens_fn

        groups = len(all_datasets) if getattr(self.args, 'sentence_batching', False) else None

        if getattr(self.args, 'separate_eval', False) and (all_datasets[0].eval or all_datasets[0].test):
            return all_datasets, all_paths
        # TODO fix handling paths for multilingual
        else:
            return (
                self.combine_datasets(all_datasets, all_paths, sort_key_fn, batch_size_fn, used_fields, groups),
                all_paths[0],
            )


@register_task('almond_multilingual')
class AlmondMultiLingual(BaseAlmondMultiLingualTask):
    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['em', 'sm', 'bleu']

    def _is_program_field(self, field_name):
        return field_name == 'answer'

    @property
    def utterance_field(self):
        return 'context'

    def _make_example(self, parts, dir_name, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, sentence, target_code = parts[:3]
        else:
            example_id, sentence, target_code = parts
        language = ISO_to_LANG.get(dir_name, 'English').lower()
        if self.args.almond_lang_as_question:
            question = 'translate from {} to thingtalk'.format(language)
        else:
            question = 'translate from english to thingtalk'
        context = sentence
        answer = target_code
        return Example.from_raw(
            self.name + '/' + dir_name + '/' + example_id,
            context,
            question,
            answer,
            preprocess=self.preprocess_field,
            lower=False,
        )


@register_task('almond_dialogue_multilingual_nlu')
class AlmondDialogMultiLingualNLU(BaseAlmondMultiLingualTask):
    """Multi-turn NLU task (user and agent) for MultiLingual Almond dialogues"""

    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['em', 'sm', 'bleu']

    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')

    @property
    def utterance_field(self):
        return 'question'

    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts
        else:
            example_id, context, sentence, target_code = parts[:4]
        answer = target_code
        question = sentence
        return Example.from_raw(
            self.name + '/' + dir_name + '/' + example_id,
            context,
            question,
            answer,
            preprocess=self.preprocess_field,
            lower=False,
        )


@register_task('almond_dialogue_multilingual_nlg')
class AlmondDialogMultiLingualNLG(BaseAlmondTask):
    """Multi-turn NLG task (agent) for MultiLingual Almond dialogues"""

    def __init__(self, name, args):
        super().__init__(name, args)
        self._metrics = ['bleu']

    def _is_program_field(self, field_name):
        return field_name == 'context'

    @property
    def utterance_field(self):
        return 'question'

    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task
        example_id, context, sentence, target_code = parts
        question = 'what should the agent say ?'
        context = context + ' ' + target_code
        answer = sentence
        return Example.from_raw(
            self.name + '/' + dir_name + '/' + example_id,
            context,
            question,
            answer,
            preprocess=self.preprocess_field,
            lower=False,
        )
