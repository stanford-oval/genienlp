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


from . import generic_dataset
import revtok


class BaseTask:
    """
    Base class for all tasks.

    Includes all the code to handle generic tasks

    """

    def __init__(self, name, args):
        self.name = name
        self.force_subword_tokenize = args.force_subword_tokenize

    @property
    def default_question(self):
        return ''

    @property
    def default_context(self):
        return ''

    def tokenize(self, sentence, field_name=None):
        if not sentence:
            return [], None
        return revtok.tokenize(sentence), None

    def detokenize(self, tokenized, field_name=None):
        return revtok.detokenize(tokenized)

    def get_splits(self, root, **kwargs):
        """
        Load the train, test, eval datasets for this task

        :param field: the text.Field to use for tokenization, preprocessing and vocabulary construction
        :param root: the base directory where data is stored
        :param kwargs: other arguments to pass to the Dataset
        :return: a list of text.Dataset
        """
        return generic_dataset.JSON.splits(root=root, name=self.name, tokenize=self.tokenize, **kwargs)

    def preprocess_example(self, ex, train=False, max_context_length=None):
        """
        Preprocess a given example, in a task specific way.

        Returns the modified example, or None if the example should be dropped from the dataset

        :param ex: the text.Example to preprocess
        :return: a new Example or None
        """
        return ex

    @property
    def metrics(self):
        """
        What metrics to evaluate this task on.

        This property must return a non-empty list.
        The first entry in the list will be the metric to use to compute the decascore.

        :return: a list of metric names
        """
        return ['em', 'nem', 'nf1']
