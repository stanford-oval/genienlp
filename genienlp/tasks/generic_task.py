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

from .base_task import BaseTask
from .registry import register_task
from . import generic_dataset


@register_task('multi30k')
class Multi30K(BaseTask):
    @property
    def metrics(self):
        return ['bleu', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        src, trg = ['.' + x for x in self.name.split('.')[1:]]
        return generic_dataset.Multi30k.splits(exts=(src, trg),
                                               root=root,
                                               tokenize=self.tokenize,
                                               **kwargs)


@register_task('iwslt')
class IWSLT(BaseTask):
    @property
    def metrics(self):
        return ['bleu', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        src, trg = ['.' + x for x in self.name.split('.')[1:]]
        return generic_dataset.IWSLT.splits(exts=(src, trg),
                                            root=root,
                                            tokenize=self.tokenize,
                                            **kwargs)


@register_task('squad')
class SQuAD(BaseTask):
    @property
    def metrics(self):
        return ['nf1', 'em', 'nem']

    def tokenize(self, sentence, field_name=None):
        if not sentence:
            return [], None
        return sentence.split(), None

    def detokenize(self, tokenized, field_name=None):
        return ' '.join(tokenized)

    def get_splits(self, root, **kwargs):
        return generic_dataset.SQuAD.splits(root=root,
                                            description=self.name,
                                            tokenize=self.tokenize,
                                            **kwargs)


@register_task('wikisql')
class WikiSQL(BaseTask):
    @property
    def metrics(self):
        return ['lfem', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        return generic_dataset.WikiSQL.splits(
            root=root,
            query_as_question='query_as_question' in self.name,
            tokenize=self.tokenize,
            **kwargs)


@register_task('ontonotes')
class OntoNotesNER(BaseTask):
    def get_splits(self, root, **kwargs):
        split_task = self.name.split('.')
        _, _, subtask, nones, counting = split_task
        return generic_dataset.OntoNotesNER.splits(
            subtask=subtask, nones=True if nones == 'nones' else False,
            root=root,
            tokenize=self.tokenize,
            **kwargs)


@register_task('woz')
class WoZ(BaseTask):
    @property
    def metrics(self):
        return ['joint_goal_em', 'turn_request_em', 'turn_goal_em', 'avg_dialogue', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        return generic_dataset.WOZ.splits(description=self.name,
                                          root=root,
                                          tokenize=self.tokenize,
                                          **kwargs)


@register_task('multinli')
class MultiNLI(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.MultiNLI.splits(description=self.name,
                                               root=root,
                                               tokenize=self.tokenize,
                                               **kwargs)


@register_task('srl')
class SRL(BaseTask):
    @property
    def metrics(self):
        return ['nf1', 'em', 'nem']

    def get_splits(self, root, **kwargs):
        return generic_dataset.SRL.splits(root=root, tokenize=self.tokenize, **kwargs)


@register_task('snli')
class SNLI(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.SNLI.splits(root=root, tokenize=self.tokenize, **kwargs)


@register_task('schema')
class WinogradSchema(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.WinogradSchema.splits(root=root, tokenize=self.tokenize, **kwargs)


class BaseSummarizationTask(BaseTask):
    @property
    def metrics(self):
        return ['avg_rouge', 'rouge1', 'rouge2', 'rougeL', 'em', 'nem', 'nf1']

    def preprocess_example(self, ex, train=False, max_context_length=None):
        # Filter examples with a dummy summary
        if train and 'This page includes the show' in ex.answer:
            return None

        return ex._replace(context=ex.context[:max_context_length])


@register_task('cnn')
class CNN(BaseSummarizationTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.CNN.splits(root=root, tokenize=self.tokenize, **kwargs)


@register_task('dailymail')
class DailyMail(BaseSummarizationTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.DailyMail.splits(root=root, tokenize=self.tokenize, **kwargs)


@register_task('cnn_dailymail')
class CNNDailyMail(BaseSummarizationTask):
    def get_splits(self, root, **kwargs):
        split_cnn = generic_dataset.CNN.splits(root=root, tokenize=self.tokenize, **kwargs)
        split_dm = generic_dataset.DailyMail.splits(root=root, tokenize=self.tokenize, **kwargs)
        for scnn, sdm in zip(split_cnn, split_dm):
            scnn.examples.extend(sdm)
        return split_cnn


@register_task('sst')
class SST(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.SST.splits(root=root, tokenize=self.tokenize, **kwargs)


@register_task('imdb')
class IMDB(BaseTask):
    def preprocess_example(self, ex, train=False, max_context_length=None):
        return ex._replace(context=ex.context[:max_context_length])

    def get_splits(self, root, **kwargs):
        kwargs['validation'] = None
        return generic_dataset.IMDb.splits(root=root, tokenize=self.tokenize, **kwargs)


@register_task('zre')
class ZRE(BaseTask):
    @property
    def metrics(self):
        return ['corpus_f1', 'precision', 'recall', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        return generic_dataset.ZeroShotRE.splits(root=root, tokenize=self.tokenize, **kwargs)
