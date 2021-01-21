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
import math
import multiprocessing as mp

from .generic_dataset import CQA
from .almond_utils import process, chunk_file

from .base_dataset import Split

logger = logging.getLogger(__name__)


class AlmondDataset(CQA):
    """Obtaining dataset for Almond semantic parsing task"""

    base_url = None

    def __init__(self, path, *, make_example, **kwargs):
        
        #TODO fix cache_path for multilingual task
        subsample = kwargs.get('subsample')
        cached_path = kwargs.get('cached_path')
        is_contextual = kwargs.get('is_contextual')
        
        skip_cache = kwargs.get('skip_cache', True)
        cache_input_data = kwargs.get('cache_input_data', False)
        bootleg = kwargs.get('bootleg', None)
        db = kwargs.get('db', None)
        DBtype2TTtype = kwargs.get('DBtype2TTtype', None)
        num_workers = kwargs.get('num_workers', 0)
        features_size = kwargs.get('features_size')
        features_default_val = kwargs.get('features_default_val')
        verbose = kwargs.get('verbose', False)
        
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
                    process_args = [{'in_file': chunk_file_paths[i], 'chunk_size': chunk_size, 'dir_name': dir_name,
                                     'example_batch_size': 1, 'make_process_example': make_example,
                                     'kwargs': kwargs} for i in range(num_chunks)]
                    results = pool.map(process, process_args)
    
                # merge all results
                examples = [item for sublist in results for item in sublist]
    
                for file in chunk_file_paths:
                    os.remove(file)
            else:
                process_args = {'in_file': path, 'chunk_size': max_examples, 'dir_name': dir_name,
                                'example_batch_size': 1, 'make_process_example': make_example,
                                'kwargs': kwargs}
                examples = process(process_args)
                
                
            if bootleg:
                config_ovrrides = bootleg.fixed_overrides
                
                input_file_dir = os.path.dirname(path)
                input_file_name = os.path.basename(path.rsplit('.', 1)[0] + '_bootleg.jsonl')
                
                data_overrides = [
                    "--data_config.data_dir", input_file_dir,
                    "--data_config.test_dataset.file", input_file_name
                ]
                
                # get config args
                config_ovrrides.extend(data_overrides)
                config_args = bootleg.create_config(config_ovrrides)
                
                # create jsonl files from input examples
                # jsonl is the input format bootleg expects
                bootleg.create_jsonl(path, examples, is_contextual)
                
                # extract mentions and mention spans in the sentence and write them to output jsonl files
                bootleg.extract_mentions(path)
                
                # find the right entity candidate for each mention
                bootleg.disambiguate_mentions(config_args)
                
                # extract features for each token in input sentence from bootleg outputs
                all_token_type_ids, all_tokens_type_probs = bootleg.collect_features(input_file_name[:-len('_bootleg.jsonl')])
                
                # override examples features with bootleg features
                assert len(examples) == len(all_token_type_ids) == len(all_tokens_type_probs)
                for n, (ex, tokens_type_ids, tokens_type_probs) in enumerate(zip(examples, all_token_type_ids, all_tokens_type_probs)):
                    if is_contextual:
                        for i in range(len(tokens_type_ids)):
                            examples[n].question_feature[i].type_id = tokens_type_ids[i]
                            examples[n].question_feature[i].type_prob = tokens_type_probs[i]
                            examples[n].context_plus_question_feature[i + len(ex.context.split(' '))].type_id = tokens_type_ids[i]
                            examples[n].context_plus_question_feature[i + len(ex.context.split(' '))].type_prob = tokens_type_probs[i]
 
                    else:
                        for i in range(len(tokens_type_ids)):
                            examples[n].context_feature[i].type_id = tokens_type_ids[i]
                            examples[n].context_feature[i].type_prob = tokens_type_probs[i]
                            examples[n].context_plus_question_feature[i].type_id = tokens_type_ids[i]
                            examples[n].context_plus_question_feature[i].type_prob = tokens_type_probs[i]


                    context_plus_question_with_types_tokens = []
                    context_plus_question_tokens = ex.context_plus_question.split(' ')
                    context_plus_question_features = ex.context_plus_question_feature
                    i = 0
                    while i < len(context_plus_question_tokens):
                        token = context_plus_question_tokens[i]
                        feat = context_plus_question_features[i]
                        # token is entity
                        if any([val != features_default_val[0] for val in feat.type_id]):
                            final_token = '<e> '
                            all_types = ' | '.join([DBtype2TTtype.get(db.id2type[id], '') for id in feat.type_id])
                            final_token += '( ' + all_types + ' ) ' + token
                            # append all entities with same type
                            i += 1
                            while i < len(context_plus_question_tokens) and context_plus_question_features[i] == feat:
                                final_token += ' ' + context_plus_question_tokens[i]
                                i += 1
                            final_token += ' </e>'
                            context_plus_question_with_types_tokens.append(final_token)
                        else:
                            context_plus_question_with_types_tokens.append(token)
                            i += 1
                    context_plus_question_with_types = ' '.join(context_plus_question_with_types_tokens)
                    examples[n] = examples[n]._replace(context_plus_question_with_types=context_plus_question_with_types)

                if verbose:
                    for ex in examples:
                        print()
                        print(*[f'token: {token}\ttype: {token_type}' for token, token_type in zip(ex.context_plus_question.split(' '), ex.context_plus_question_feature)], sep='\n')

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
        
        return Split(train=None if train is None else train_data,
                     eval=None if validation is None else validation_data,
                     test=None if test is None else test_data,
                     aux=None if do_curriculum is None else aux_data)
    

