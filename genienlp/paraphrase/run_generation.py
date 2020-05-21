# Parts of this file were adopted from https://github.com/huggingface/transformers.
# See the original copyright notice below.

# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with GPT-2/BART
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from tqdm import tqdm
import torch
import math
import json
import re
import copy
import os


# multiprocessing with CUDA
from torch.multiprocessing import Process, set_start_method

from genienlp.paraphrase.data_utils import create_features_from_tsv_file, output_heuristics
from genienlp.paraphrase.model_utils import compute_metrics

try:
     set_start_method('spawn')
except RuntimeError:
    pass
 
import torch

from transformers import GPT2Config, BartConfig

from transformers import GPT2Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer, MBartTokenizer
from transformers import PretrainedConfig
from ..util import set_seed, combine_files_on_disk, split_file_on_disk, get_part_path
from .GPT2Seq2Seq import GPT2Seq2Seq


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, BartConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2Seq2Seq, GPT2Tokenizer, {'sep_token': '<paraphrase>', 'end_token': '</paraphrase>'}),
    'bart': (BartForConditionalGeneration, BartTokenizer, {'sep_token': '<s>', 'end_token': '</s>'}), # sep_token will not be used for BART
    'mbart': (BartForConditionalGeneration, MBartTokenizer, {'sep_token': '<s>', 'end_token': '</s>'}) # sep_token will not be used for BART
}


def parse_argv(parser):
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--input_file", type=str, help="The file from which we read prompts. Defaults to stdin.")
    parser.add_argument('--input_column', type=int, required=True,
                        help='The column in the input file which contains the input sentences.')
    parser.add_argument('--prompt_column', type=int, default=None,
                        help='The column in the input file which contains the text we should start generation from.')
    parser.add_argument('--gold_column', type=int, default=None,
                        help='The column in the input file which contains the gold sentences. Defaults to --input_column if no gold is available.')
    parser.add_argument('--thingtalk_column', type=int, default=None,
                        help='The column in the input file which contains the ThingTalk program.')
    parser.add_argument("--output_file", type=str, help="When specified, generated text will be written in this file. Defaults to stdout.")

    parser.add_argument('--output_prompt', action='store_true',
                        help='Whether we should include the prompt (specified via --prompt_column or --copy) in the output sequence')
    parser.add_argument("--length", type=int, default=20, help='The generated sentences will have a maximum length of len(input) + arg.length')
    parser.add_argument("--min_output_length", type=int, default=2, help='Will prevent stop tokens from appearing in the first --min_output_length tokens of the generated sentences.')
    parser.add_argument("--skip_heuristics", action='store_true', help='If True, will not replace special word such as NUMBER_0 in the input.')
    parser.add_argument("--is_cased", action='store_true',
                        help='If True, the trained model is cased, so if --skip_heuristics is not set, we will convert the input to upper case and the output back to lower case.')
    parser.add_argument("--metric_reduction", type=str, choices=['average', 'max'], default='average',
                        help="How we should calculate metrics where there are multiple generations per example.")
    

    # These are generation hyperparameters. Each one can be a list of values in which case, we generate num_samples outputs for each set of hyperparameters.
    parser.add_argument("--num_samples", type=int, nargs='+', default=[1])
    parser.add_argument("--temperature", type=float, nargs='+', default=[1.0],
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, nargs='+', default=[1.0],
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[0.9], help='1.0 disables top-p filtering')
    parser.add_argument("--num_beams", type=int, nargs='+', default=[1], help='1 disables beam seach')
    parser.add_argument("--no_repeat_ngram_size", type=int, nargs='+', default=[0], help='ngrams of this size cannot be repeated in the output. 0 disables it.')
    
    parser.add_argument("--copy", type=int, default=0,
                        help='Number of tokens that will be copied at the beginning of generation. Helps preserve the original meaning of the input sequence.')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_tokens', type=str, nargs='+', default=[],
                        help="Tokens (other than the model-specific `end_token`) at which text generation should be stopped.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for text generation for each GPU.")

def main(args):
    config = PretrainedConfig.from_pretrained(args.model_name_or_path)
    
    # get model type from saved config
    if hasattr(config, 'model_type'):
        args.model_type = getattr(config, 'model_type')
        
        # bart and mbart share the same config
        # check which model we are actually using
        if args.model_type == 'bart':
            try:
                if config.normalize_before and config.add_final_layer_norm and config.scale_embedding:
                    args.model_type = 'mbart'
            except AttributeError as e:
                args.model_type = 'bart'
            
    else:
        raise ValueError('Model should be either GPT2, BART, or MBART')

    if args.prompt_column is not None and args.copy is not None and args.copy != 0:
        raise ValueError('Cannot copy from the input and use prompt at the same time. Disable either --copy or --prompt_column.')
    hyperparameters = ['num_samples', 'temperature', 'top_k', 'top_p', 'repetition_penalty', 'num_beams', 'no_repeat_ngram_size']
    max_hyperparameter_len = max([len(getattr(args, h)) for h in hyperparameters])
    valid_len = [1, max_hyperparameter_len]
    for h in hyperparameters:
        if (len(getattr(args, h)) not in valid_len):
            logger.error('Hyperparameters should either have the same number of values as others or have exactly one value.')
        # If only one value is provided, use the same value for all samples
        setattr(args, h, getattr(args, h) * (max_hyperparameter_len // len(getattr(args, h))))

    logger.info('Will output %d sequences for each input.', sum(args.num_samples))
    logger.info('Effective batch size for each GPU is %d', args.batch_size*max(args.num_samples))

    if args.gold_column is None:
        args.gold_column = args.input_column
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    if args.n_gpu > 1:
        if args.input_file is None:
            raise ValueError('Cannot use multiple GPUs when reading from stdin. You should provide an --input_file')
        # Independent multi-GPU generation
        all_processes = []
        all_input_files = split_file_on_disk(args.input_file, args.n_gpu)
        for gpu_idx in range(args.n_gpu):
            copy_args = copy.copy(args)
            if torch.cuda.is_available() and not args.no_cuda:
                copy_args.device = torch.device("cuda:" + str(gpu_idx))
            copy_args.n_gpu = 1
            copy_args.input_file = all_input_files[gpu_idx]
            copy_args.output_file = get_part_path(args.output_file, gpu_idx)
            
            p = Process(target=run_generation, args=(copy_args,))
            all_processes.append(p)
            p.start()

        for p in all_processes:
            p.join()

        for file in all_input_files:
            os.remove(file)
        combine_files_on_disk(args.output_file, args.n_gpu, line_group_size=sum(args.num_samples), delete=True)

    else:
        run_generation(args)


def run_generation(args):
    model_class, tokenizer_class, special_tokens = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    end_token_id = tokenizer.convert_tokens_to_ids(special_tokens['end_token'])
    sep_token_id = tokenizer.convert_tokens_to_ids(special_tokens['sep_token'])
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    if pad_token_id is None:
        logger.error('Your tokenizer does not have a padding token')

    if args.model_type == 'gpt2':
        model.set_token_ids(end_token_id=end_token_id, 
                            sep_token_id=sep_token_id, 
                            pad_token_id=pad_token_id)

    logger.info(args)

    all_input_sequences, all_input_sequence_lengths, all_context_tokens, estimated_output_lengths, all_golds, reverse_maps, all_prompt_tokens = \
                                  create_features_from_tsv_file(file_path=args.input_file, tokenizer=tokenizer,
                                                                input_column=args.input_column, gold_column=args.gold_column, prompt_column=args.prompt_column,
                                                                copy=args.copy,
                                                                thingtalk_column=args.thingtalk_column,
                                                                sep_token_id=sep_token_id, skip_heuristics=args.skip_heuristics, is_cased=args.is_cased,
                                                                model_type=args.model_type)

    # sort contexts based on their context length so that less generated tokens are thrown away and generation can be done faster
    estimated_output_lengths, all_input_sequence_lengths, all_input_sequences, all_context_tokens, original_order, reverse_maps, all_prompt_tokens = \
        tuple(zip(*sorted(list(zip(estimated_output_lengths, all_input_sequence_lengths, all_input_sequences, all_context_tokens, range(len(all_context_tokens)), reverse_maps, all_prompt_tokens)), reverse=True)))
    all_outputs = []

    stop_token_ids = [tokenizer.convert_tokens_to_ids(stop_token) for stop_token in args.stop_tokens]
    
    batch_idx = 0
    for batch in tqdm(range(math.ceil(len(all_context_tokens) / args.batch_size)), desc="Batch"):
        logging.info('') # to make kubectl properly print tqdm progress bar
        batch_slice = (batch*args.batch_size, min((batch+1)*args.batch_size, len(all_context_tokens)))
        batch_size = batch_slice[1] - batch_slice[0]
        batch_input_sequences = all_input_sequences[batch_slice[0]: batch_slice[1]]
        batch_input_sequence_lengths = all_input_sequence_lengths[batch_slice[0]: batch_slice[1]]
        batch_context_tokens = all_context_tokens[batch_slice[0]: batch_slice[1]]
        batch_reverse_maps = reverse_maps[batch_slice[0]: batch_slice[1]]
        batch_prompt_tokens = all_prompt_tokens[batch_slice[0]: batch_slice[1]]

        if args.model_type == 'gpt2':
            batch_context_tensor = torch.tensor(model.pad_to_max_length(batch_context_tokens), dtype=torch.long, device=args.device)
            attention_mask = None
        elif args.model_type == 'bart' or args.model_type == 'mbart':
            padded_batch_context_tokens = []
            max_length = max([len(s) for s in batch_context_tokens])
            for i in range(len(batch_context_tokens)):
                padded_batch_context_tokens.append(batch_context_tokens[i]+[pad_token_id]*(max_length-len(batch_context_tokens[i])))
            batch_context_tensor = torch.tensor(padded_batch_context_tokens, dtype=torch.long, device=args.device)
            attention_mask = (batch_context_tensor!=pad_token_id).to(torch.long)

        batch_outputs = [[] for _ in range(batch_size)]
        for hyperparameter_idx in range(len(args.temperature)):
            out = model.generate(input_ids=batch_context_tensor,
                                 bad_words_ids=None,
                                 attention_mask=attention_mask,
                                 min_length=args.min_output_length,
                                 max_length=batch_context_tensor.shape[1]+args.length,
                                 num_beams=args.num_beams[hyperparameter_idx],
                                 top_k=args.top_k[hyperparameter_idx],
                                 top_p=args.top_p[hyperparameter_idx],
                                 early_stopping=True,
                                 num_return_sequences=args.num_samples[hyperparameter_idx],
                                 repetition_penalty=args.repetition_penalty[hyperparameter_idx],
                                 no_repeat_ngram_size=args.no_repeat_ngram_size[hyperparameter_idx],
                                 do_sample=args.temperature[hyperparameter_idx]!=0,
                                 temperature=args.temperature[hyperparameter_idx] if args.temperature[hyperparameter_idx] > 0 else 1.0, # if temperature==0, we do not sample
                                 eos_token_id=end_token_id,
                                 pad_token_id=pad_token_id,
                                )
            if not isinstance(out, list):
                out = out[:, :].tolist()
            for i, o in enumerate(out):
                if args.model_type=='bart' or args.model_type=='mbart':
                    o = o[1:] # remove <s> start token
                if not args.output_prompt:
                    o = o[len(batch_prompt_tokens[(i//args.num_samples[hyperparameter_idx]) % batch_size]):]
                min_index = len(o)-1
                for stop_token_id in stop_token_ids+[end_token_id]:
                    try:
                        index = o.index(stop_token_id)
                        min_index = min(index, min_index)
                    except ValueError:
                        pass
                if o[min_index] != end_token_id:
                    min_index = min_index + 1 # include the last token if it is not end_token
                o = o[:min_index]
                
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True, skip_special_tokens=True)

                text = re.sub('\s\s+', ' ', text) # remove duplicate white spaces
                text = text.strip()
                if not args.skip_heuristics:
                    text = output_heuristics(text, batch_reverse_maps[(i//args.num_samples[hyperparameter_idx]) % batch_size])
                batch_outputs[(i//args.num_samples[hyperparameter_idx]) % batch_size].append(text)

        all_outputs.extend(batch_outputs)
        if batch_idx < 1:
            logger.info('First batch output: %s', str(all_outputs))
            batch_idx += 1


    # sort the results back to their original order
    _, all_outputs = tuple(zip(*sorted(list(zip(original_order, all_outputs)))))

    if args.output_file is not None:
        if not os.path.exists(os.path.dirname(args.output_file)):
            os.makedirs(os.path.dirname(args.output_file), exist_ok=False)
        with open(args.output_file, 'w') as output_file:
            for output in all_outputs:
                for text in output:
                    output_file.write(text + '\n')
    else:
        print(json.dumps(all_outputs, indent=2))

    metrics = compute_metrics(all_outputs, all_golds, reduction=args.metric_reduction)
    logger.info('Average BLEU score = %.2f', metrics['bleu'])
    logger.info('Exact match score = %.2f', metrics['em'])

