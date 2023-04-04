# Copyright 2019 The Board of Trustees of the Leland Stanford Junior University
#
# Author: Giovanni Campagna <gcampagn@cs.stanford.edu>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
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

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='genienlp',
    version='0.7.0a4',
    packages=setuptools.find_packages(exclude=['tests']),
    package_data={'genienlp': ['ned/database_files/*']},
    entry_points={
        'console_scripts': ['genienlp=genienlp.__main__:main'],
    },
    license='BSD-3-Clause',
    author="Salesforce Inc., Stanford University Open Virtual Assistant Lab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stanford-oval/genienlp",
    install_requires=[
        'numpy>=1.14.5',
        'torch>=1.9.0,<1.14.0',
        'transformers==4.22.2',
        'datasets==2.11.0',
        'pathos==0.2.8',
        'protobuf==3.20.3',
        # for sts:
        'sentence-transformers==2.2.2',
        # for alignment:
        'num2words>=0.5.10',
        'dateparser>=1.0.0',
        # for metrics:
        'rouge_score>=0.0.4',
        'sacrebleu>=1,<3',
        'bert-score~=0.3',
        'seqeval==1.2.2',
        # for kf:
        'kfserving>=0.5.0',
        'ray==1.6.0',
        # for NED:
        'bootleg==1.0.5',
        # for calibration:
        'scikit-learn>=0.23,<2.0',
        'dill~=0.3',
        'xgboost~=1.3',
        # for e2e dialogues:
        'dialogues==0.0.3',
    ],
)
