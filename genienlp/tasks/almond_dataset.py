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
import math
import multiprocessing as mp
import os

import torch

from ..data_utils.almond_utils import chunk_file, create_examples_from_file
from .base_dataset import Split
from .generic_dataset import CQA

logger = logging.getLogger(__name__)


class AlmondDataset(CQA):
    """Obtaining dataset for Almond semantic parsing task"""

    base_url = None

    def __init__(self, path, *, make_example, **kwargs):

        # TODO fix cache_path for multilingual task
        subsample = kwargs.get('subsample')
        cached_path = kwargs.get('cached_path')

        skip_cache = kwargs.get('skip_cache', True)
        cache_input_data = kwargs.get('cache_input_data', False)
        num_workers = kwargs.get('num_workers', 0)

        cache_name = os.path.join(cached_path, os.path.basename(path), str(subsample))
        dir_name = os.path.basename(os.path.dirname(path))

        if os.path.exists(cache_name) and not skip_cache:
            logger.info(f'Loading cached data from {cache_name}')
            examples = torch.load(cache_name)
        else:
            n = 0
            with open(path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    n += 1

            max_examples = min(n, subsample) if subsample is not None else n
            if num_workers > 0:
                num_processes = min(num_workers, int(mp.cpu_count()))
                logger.info(f'Using {num_processes} workers...')
                chunk_size = int(math.ceil(max_examples / num_processes))
                num_chunks = int(math.ceil(max_examples / chunk_size))

                base_path, extension = path.rsplit('.', 1)

                chunk_file_paths = [f'{base_path}_{chunk_id}.tsv' for chunk_id in range(num_chunks)]
                chunk_file(path, chunk_file_paths, chunk_size, num_chunks)
                num_processes = min(num_processes, num_chunks)

                with mp.Pool(processes=num_processes) as pool:
                    process_args = [
                        {
                            'in_file': chunk_file_paths[i],
                            'chunk_size': chunk_size,
                            'dir_name': dir_name,
                            'example_batch_size': 1,
                            'make_process_example': make_example,
                            'kwargs': kwargs,
                        }
                        for i in range(num_chunks)
                    ]
                    results = pool.map(create_examples_from_file, process_args)

                # merge all results
                examples = [item for sublist in results for item in sublist]

                for file in chunk_file_paths:
                    os.remove(file)
            else:
                process_args = {
                    'in_file': path,
                    'chunk_size': max_examples,
                    'dir_name': dir_name,
                    'example_batch_size': 1,
                    'make_process_example': make_example,
                    'kwargs': kwargs,
                }
                examples = create_examples_from_file(process_args)

            if cache_input_data:
                os.makedirs(os.path.dirname(cache_name), exist_ok=True)
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

        data_splits = Split(
            train=None if train is None else train_data,
            eval=None if validation is None else validation_data,
            test=None if test is None else test_data,
            aux=None if do_curriculum is False else aux_data,
        )

        all_paths = Split(
            train=None if train is None else os.path.join(path, train + '.tsv'),
            eval=None if validation is None else os.path.join(path, validation + '.tsv'),
            test=None if test is None else os.path.join(path, test + '.tsv'),
            aux=None if do_curriculum is False else os.path.join(path, 'aux' + '.tsv'),
        )

        return data_splits, all_paths
