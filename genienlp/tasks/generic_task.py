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

from collections import OrderedDict

from ..data_utils.example import Example
from . import generic_dataset
from .almond_task import BaseAlmondTask
from .base_task import BaseTask
from .generic_dataset import CrossNERDataset, OODDataset
from .registry import register_task


@register_task('multi30k')
class Multi30K(BaseTask):
    @property
    def metrics(self):
        return ['bleu', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        src, trg = ['.' + x for x in self.name.split('.')[1:]]
        return generic_dataset.Multi30k.splits(exts=(src, trg), root=root, **kwargs)


@register_task('iwslt')
class IWSLT(BaseTask):
    @property
    def metrics(self):
        return ['bleu', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        src, trg = ['.' + x for x in self.name.split('.')[1:]]
        return generic_dataset.IWSLT.splits(exts=(src, trg), root=root, **kwargs)


@register_task('squad')
class SQuAD(BaseTask):
    @property
    def metrics(self):
        return ['nf1', 'em', 'nem']

    def get_splits(self, root, **kwargs):
        return generic_dataset.SQuAD.splits(root=root, description=self.name, **kwargs)


@register_task('wikisql')
class WikiSQL(BaseTask):
    @property
    def metrics(self):
        return ['lfem', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        return generic_dataset.WikiSQL.splits(root=root, query_as_question='query_as_question' in self.name, **kwargs)


@register_task('ontonotes')
class OntoNotesNER(BaseTask):
    def get_splits(self, root, **kwargs):
        split_task = self.name.split('.')
        _, _, subtask, nones, counting = split_task
        return generic_dataset.OntoNotesNER.splits(
            subtask=subtask, nones=True if nones == 'nones' else False, root=root, **kwargs
        )


@register_task('woz')
class WoZ(BaseTask):
    @property
    def metrics(self):
        return ['joint_goal_em', 'turn_request_em', 'turn_goal_em', 'avg_dialogue', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        return generic_dataset.WOZ.splits(description=self.name, root=root, **kwargs)


@register_task('multinli')
class MultiNLI(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.MultiNLI.splits(description=self.name, root=root, **kwargs)


@register_task('srl')
class SRL(BaseTask):
    @property
    def metrics(self):
        return ['nf1', 'em', 'nem']

    def get_splits(self, root, **kwargs):
        return generic_dataset.SRL.splits(root=root, **kwargs)


@register_task('snli')
class SNLI(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.SNLI.splits(root=root, **kwargs)


@register_task('schema')
class WinogradSchema(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.WinogradSchema.splits(root=root, **kwargs)


class BaseSummarizationTask(BaseTask):
    @property
    def metrics(self):
        return ['avg_rouge', 'rouge1', 'rouge2', 'rougeL', 'em', 'nem', 'nf1']


@register_task('cnn')
class CNN(BaseSummarizationTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.CNN.splits(root=root, **kwargs)


@register_task('dailymail')
class DailyMail(BaseSummarizationTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.DailyMail.splits(root=root, **kwargs)


@register_task('cnn_dailymail')
class CNNDailyMail(BaseSummarizationTask):
    def get_splits(self, root, **kwargs):
        split_cnn = generic_dataset.CNN.splits(root=root, **kwargs)
        split_dm = generic_dataset.DailyMail.splits(root=root, **kwargs)
        for scnn, sdm in zip(split_cnn, split_dm):
            scnn.examples.extend(sdm)
        return split_cnn


@register_task('sst')
class SST(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.SST.splits(root=root, **kwargs)


@register_task('imdb')
class IMDB(BaseTask):
    def get_splits(self, root, **kwargs):
        kwargs['validation'] = None
        return generic_dataset.IMDb.splits(root=root, **kwargs)


@register_task('zre')
class ZRE(BaseTask):
    @property
    def metrics(self):
        return ['corpus_f1', 'precision', 'recall', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        return generic_dataset.ZeroShotRE.splits(root=root, **kwargs)


@register_task('cross_ner')
class CrossNERTask(BaseAlmondTask):
    politics_labels = [
        'O',
        'B-country',
        'B-politician',
        'I-politician',
        'B-election',
        'I-election',
        'B-person',
        'I-person',
        'B-organisation',
        'I-organisation',
        'B-location',
        'B-misc',
        'I-location',
        'I-country',
        'I-misc',
        'B-politicalparty',
        'I-politicalparty',
        'B-event',
        'I-event',
    ]

    science_labels = [
        'O',
        'B-scientist',
        'I-scientist',
        'B-person',
        'I-person',
        'B-university',
        'I-university',
        'B-organisation',
        'I-organisation',
        'B-country',
        'I-country',
        'B-location',
        'I-location',
        'B-discipline',
        'I-discipline',
        'B-enzyme',
        'I-enzyme',
        'B-protein',
        'I-protein',
        'B-chemicalelement',
        'I-chemicalelement',
        'B-chemicalcompound',
        'I-chemicalcompound',
        'B-astronomicalobject',
        'I-astronomicalobject',
        'B-academicjournal',
        'I-academicjournal',
        'B-event',
        'I-event',
        'B-theory',
        'I-theory',
        'B-award',
        'I-award',
        'B-misc',
        'I-misc',
    ]

    music_labels = [
        'O',
        'B-musicgenre',
        'I-musicgenre',
        'B-song',
        'I-song',
        'B-band',
        'I-band',
        'B-album',
        'I-album',
        'B-musicalartist',
        'I-musicalartist',
        'B-musicalinstrument',
        'I-musicalinstrument',
        'B-award',
        'I-award',
        'B-event',
        'I-event',
        'B-country',
        'I-country',
        'B-location',
        'I-location',
        'B-organisation',
        'I-organisation',
        'B-person',
        'I-person',
        'B-misc',
        'I-misc',
    ]

    literature_labels = [
        'O',
        'B-book',
        'I-book',
        'B-writer',
        'I-writer',
        'B-award',
        'I-award',
        'B-poem',
        'I-poem',
        'B-event',
        'I-event',
        'B-magazine',
        'I-magazine',
        'B-literarygenre',
        'I-literarygenre',
        'B-country',
        'I-country',
        'B-person',
        'I-person',
        'B-location',
        'I-location',
        'B-organisation',
        'I-organisation',
        'B-misc',
        'I-misc',
    ]

    ai_labels = [
        'O',
        'B-field',
        'I-field',
        'B-task',
        'I-task',
        'B-product',
        'I-product',
        'B-algorithm',
        'I-algorithm',
        'B-researcher',
        'I-researcher',
        'B-metrics',
        'I-metrics',
        'B-programlang',
        'I-programlang',
        'B-conference',
        'I-conference',
        'B-university',
        'I-university',
        'B-country',
        'I-country',
        'B-person',
        'I-person',
        'B-organisation',
        'I-organisation',
        'B-location',
        'I-location',
        'B-misc',
        'I-misc',
    ]

    news_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    domain2labels = OrderedDict(
        {
            'politics': politics_labels,
            'science': science_labels,
            'music': music_labels,
            'literature': literature_labels,
            'ai': ai_labels,
            'news': news_labels,
        }
    )

    def __init__(self, name, args):
        self.all_labels = []
        # OrderedDict respect keys order
        for domain, labels in self.domain2labels.items():
            self.all_labels.extend(labels)
        self.label2id = {}
        for label in self.all_labels:
            if label not in self.label2id:
                self.label2id[label] = len(self.label2id)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)
        super().__init__(name, args)

    @property
    def metrics(self):
        return ['ner_f1', 'em', 'f1', 'pem']

    def _is_program_field(self, field_name):
        return field_name == 'answer'

    def utterance_field(self):
        return 'context'

    def _make_example(self, parts, dir_name=None, **kwargs):
        example_id = parts[0]
        context = ' '.join(parts[1])
        question = ''
        answer = ' '.join([str(self.label2id[label]) for label in parts[2]])

        return Example.from_raw(
            self.name + '/' + str(example_id), context, question, answer, preprocess=self.preprocess_field, lower=False
        )

    def get_splits(self, root, **kwargs):
        return CrossNERDataset.return_splits(name=self.name, path=root, make_example=self._make_example, **kwargs)


@register_task('ood_task')
class OODTask(BaseTask):
    def __init__(self, name, args):
        self.id2label = ['0', '1']
        self.num_labels = 2
        super().__init__(name, args)

    @property
    def metrics(self):
        return ['sc_f1', 'sc_precision', 'sc_recall']

    def get_splits(self, root, **kwargs):
        return OODDataset.splits(root=root, **kwargs)
