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
from tqdm import trange, tqdm
import math
import json
import csv
import re
import copy
import numpy as np
import os
import sys

# multiprocessing with CUDA
from torch.multiprocessing import Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
 
import torch

from transformers import GPT2Config, BartConfig

from transformers import GPT2Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import PretrainedConfig
from .util import set_seed, get_number_of_lines, combine_files_on_disk, split_file_on_disk, get_part_path, detokenize, tokenize, lower_case, \
                    SpecialTokenMap, remove_thingtalk_quotes
from .metrics import computeBLEU
from .GPT2Seq2Seq import GPT2Seq2Seq


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, BartConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2Seq2Seq, GPT2Tokenizer, {'sep_token': '<paraphrase>', 'end_token': '</paraphrase>'}),
    'bart': (BartForConditionalGeneration, BartTokenizer, {'sep_token': '<s>', 'end_token': '</s>'}) # sep_token will not be used for BART
}


special_pattern_mapping = [
    SpecialTokenMap('PHONE_NUMBER_([0-9]+)', ['888-8888', '777-8888']),
    SpecialTokenMap('NUMBER_([0-9]+)', ['2', '3'], [['2', 'two'], ['3', 'three']]),
    SpecialTokenMap('PATH_NAME_([0-9]+)', ['my1folder', 'my2folder']),
    SpecialTokenMap('TIME_([0-9]+)', ['1p.m.', '2p.m.'], [['1 pm', '1pm', '1:00 pm', '1:00pm', '1p.m.', '1 p.m.', '1:00 p.m.', '1:00', 'one o\'clock', 'one'],
                                                            ['2 pm', '2pm', '2:00 pm', '2:00pm', '2p.m.', '2 p.m.', '2:00 p.m.', '2:00', 'two o\'clock', 'two']]),
    SpecialTokenMap('EMAIL_ADDRESS_([0-9]+)', ['e1@example.com', 'e2@example.com']),
    SpecialTokenMap('URL_([0-9]+)', ['my1site.com', 'my2site.com']),
    SpecialTokenMap('DATE_([0-9]+)', ['5-6-2015', '8-3-2016']),
    SpecialTokenMap('CURRENCY_([0-9]+)', ['$12', '$13'], [['$12', 'twelve dollars', '12 dollars', '$ 12', '$ 12.00', '12.00', '12'], 
                                                          ['$13', 'thirteen dollars', '13 dollars', '$ 13', '$ 13.00', '13.00', '13']]),
    SpecialTokenMap('DURATION_([0-9]+)', ['5 weeks', '6 weeks'], [['5 weeks', 'five weeks'], ['6 weeks', 'six weeks']]),
    SpecialTokenMap('LOCATION_([0-9]+)', ['locatio1n', 'locatio2n'], [['locatio1n', 'locat1n'], ['locatio2n', 'locat2n']]),
    SpecialTokenMap('QUOTED_STRING_([0-9]+)', lambda x: 'Chinese', lambda x: ['Chinese', 'chinese']), # TODO change to be more general than cuisine
    SpecialTokenMap('GENERIC_ENTITY_uk.ac.cam.multiwoz.Restaurant:Restaurant_([0-9]+)', ["restaurant1", "restaurant2", "restaurant3"]) # TODO the only reason we can get away with this unnatural replacement is that actual backward is not going to be called for this
]

def create_features_from_tsv_file(file_path, tokenizer, input_column, gold_column, prompt_column, copy, thingtalk_column, sep_token,
                                  skip_heuristics, is_cased, model_type):
    """
    Read a tsv file (this includes a text file with one example per line) and returns input features that the model needs
    Outputs:

    """
    all_input_sequences = []
    all_input_sequence_lengths = []
    all_context_tokens = []
    estimated_output_lengths = []
    all_golds = []
    reverse_maps = []

    if file_path is not None:
        number_of_lines = get_number_of_lines(file_path)
        disable_tqdm = False
        input_file = open(file_path)
    else:
        number_of_lines = 1
        disable_tqdm = True
        input_file = sys.stdin


    for line in tqdm(input_file, desc='Reading Input File', total=number_of_lines, disable=disable_tqdm):
        row = [r.strip() for r in line.split('\t')]
        input_sequence = row[input_column]
        gold = row[gold_column]
        # logger.info('gold = %s', gold)
        if not skip_heuristics:
            gold, _ = input_heuristics(gold, None, is_cased, keep_special_tokens=True, keep_tokenized=True)
        # logger.info('gold = %s', gold)
        all_golds.append(gold)
        # logger.info('before text = %s', input_sequence)
        if skip_heuristics:
            reverse_maps.append({})
        else:
            thingtalk = row[thingtalk_column] if thingtalk_column is not None else None
            # logger.info('input_sequence = %s', input_sequence)
            input_sequence, reverse_map = input_heuristics(input_sequence, thingtalk, is_cased)
            # logger.info('input_sequence = %s', input_sequence)
            reverse_maps.append(reverse_map)
        input_sequence_tokens = tokenizer.encode(input_sequence, add_special_tokens=True)
        
        prompt_tokens = [] # includes the first few tokens of the output
        if prompt_column is not None and len(row) > prompt_column:
            prompt = row[prompt_column]
            if not skip_heuristics:
                prompt, _ = input_heuristics(prompt, thingtalk, is_cased)
                # logger.info('prompt = %s', prompt)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if copy > 0:
            assert len(prompt_tokens) == 0
            prompt_tokens = context_tokens[0 : min(copy, len(context_tokens)-1)] # -1 since we should not copy sep_token
        context_tokens = input_sequence_tokens + [tokenizer.convert_tokens_to_ids(sep_token)] + prompt_tokens
        all_input_sequences.append(input_sequence)
        all_input_sequence_lengths.append(len(input_sequence_tokens))
        all_context_tokens.append(context_tokens)
        estimated_output_lengths.append(len(input_sequence_tokens)-len(prompt_tokens))
    
    if file_path is not None:
        input_file.close()

    return all_input_sequences, all_input_sequence_lengths, all_context_tokens, estimated_output_lengths, all_golds, reverse_maps

def is_question(sentence: str):
    question_words = ['which', 'what', 'where', 'how', 'who', 'when', 'is', 'are', 'am', \
                      'can', 'could', 'would', 'will', 'have', 'did', 'do', 'does', 'no is', 'yes is']
    for w in question_words:
        if sentence.startswith(w+' '):
            return True
    return False

def input_heuristics(s: str, thingtalk=None, is_cased=False, keep_special_tokens=False, keep_tokenized=False):
    """
    Changes the input string so that it is closer to what the pre-trained language models have seen during their training.
    Outputs:
        s: the new string
        reverse_map: a list of special tokens. Can be used to recover the original special_tokens in the string
    """
    reverse_map = []
    s = s.strip()
    s = tokenize(s)

    # Put question mark at the end whenever necessary.
    sentences = [sentence.strip() for sentence in re.split('\s+([.?!:])\s*', s) if len(sentence) > 0]
    # logger.info('sentences = %s', sentences)
    for idx in range(len(sentences)):
        if sentences[idx] in ['.', '?' , '!', ':']:
            continue
        if idx == len(sentences)-1 or sentences[idx+1] not in ['.', '?', '!', ':']:
            # add the missing punctuation
            if is_question(sentences[idx]):
                sentences[idx] = sentences[idx] + '?'
            else:
                sentences[idx] = sentences[idx] + '.'
        else:
            if is_question(sentences[idx]):
                assert sentences[idx+1] in ['.', '?', '!', ':']
                sentences[idx+1] = '?'

        if is_cased:
            # capitalize the first word and parameters
            if thingtalk:
                _, parameters = remove_thingtalk_quotes(thingtalk)
                # logger.info('parameters = ', parameters)
                for p in parameters:
                    capitalized_p = ' '.join([t[0].upper()+t[1:] for t in p.split()])
                    sentences[idx] = sentences[idx].replace(p, capitalized_p)
            sentences[idx] = sentences[idx].replace(' i ', ' I ')
            sentences[idx] = sentences[idx][0].upper()+sentences[idx][1:]
            
    s = ' '.join(sentences)
    if not keep_tokenized:
        s = detokenize(s)
    
    if not is_cased:
        s = lower_case(s)

    # replace special tokens with natural-looking exmaples
    reverse_map = []
    if not keep_special_tokens:
        for spm in special_pattern_mapping:
            s, r = spm.forwad(s)
            reverse_map.extend(r)

    # logger.info('s = ', s)
    return s, reverse_map

def output_heuristics(s: str, reverse_map: list):
    for spm, occurance in reverse_map:
        s = spm.backward(s, occurance)

    s = tokenize(s)
    s = lower_case(s)
    return s


def compute_metrics(generations, golds, reduction='average'):
    """
    Inputs:
        generations: a list of list of strings; generations[i] is a list of all generated outputs of the model for example i
        golds: a list of strings; golds[i] is the gold answer for example i
        reduction: how we should compute an example's metrics from its multiple generations
    """
    total_bleu = 0.0
    # all_bleu = []
    total_exact_match = 0.0
    count = 0.0
    for idx, output in enumerate(generations):
        bleu_score = 0.0
        exact_match = 0.0
        for sample in output:
            if reduction == 'average':
                bleu_score += computeBLEU([sample], [[golds[idx]]])
            else:
                bleu_score = max(bleu_score, computeBLEU([sample], [[golds[idx]]]))
            if re.sub('\s+', '', sample).lower() == re.sub('\s+', '', golds[idx]).lower():
                if reduction == 'average':
                    exact_match += 1
                else:
                    exact_match = max(exact_match, 1)
            # all_bleu.append(bleu_score)
        if reduction == 'average':
            bleu_score /= len(output)
            exact_match /= len(output)
        total_bleu += bleu_score
        total_exact_match += exact_match
        count += 1

    # from matplotlib import pyplot as plt
    # import numpy as np
    # h, b = np.histogram(all_bleu, bins=list(range(0, 105, 5)))
    # logger.info('all_bleu = ', all_bleu)
    # logger.info('h = ', h)
    # logger.info('b = ', b)
    # h = h / np.sum(h)
    # logger.info('h = ', h)
    # plt.title('GPT2 (temp=0, penalty=1.0) Paraphrases for restaurants')
    # plt.xlabel('BLEU with original')
    # plt.ylim((0.0, 1.0))
    # center = (b[:-1] + b[1:]) / 2
    # plt.bar(center, h, align='center', width=(b[1]-b[0]))
    # plt.savefig('./fig.png')

    return {'bleu': total_bleu/count, 'em': total_exact_match/count*100}

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
    parser.add_argument("--length", type=int, default=20, help='The generated sentences will have a maximum length of len(input) + arg.length')
    parser.add_argument("--min_output_length", type=int, default=1, help='Will prevent stop tokens from appearing in the first --min_length tokens of the generated sentences.')
    parser.add_argument("--skip_heuristics", action='store_true', help='If True, will not replace special word such as NUMBER_0 in the input.')
    parser.add_argument("--is_cased", action='store_true',
                        help='If True, the trained model is cased, so if --skip_heuristics is not set, we will convert the input to upper case and the output back to lower case.')
    parser.add_argument("--metric_reduction", type=str, choices=['average', 'max'], default='average',
                        help="How we should calculate metrics where there are multiple generations per example.")
    
    parser.add_argument("--num_samples", type=int, default=1)

    # These are generation hyperparameters. Each one can be a list of values in which case, we generate num_samples outputs for each set of hyperparameters.
    parser.add_argument("--temperature", type=float, nargs='+', default=[1.0],
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, nargs='+', default=[1.0],
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[0.9], help='1.0 disables top-p filtering')
    parser.add_argument("--num_beams", type=int, nargs='+', default=[1], help='1 disables beam seach')

    parser.add_argument("--copy", type=int, default=0,
                        help='Number of tokens that will be copied at the beginning of generation. Helps preserve the original meaning of the input sequence.')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_tokens', type=str, nargs='+', default=[],
                        help="Token at which text generation is stopped.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for text generation for each GPU.")

def main(args):
    config = PretrainedConfig.from_pretrained(args.model_name_or_path)
    if config.architectures[0] == 'BartForConditionalGeneration':
        args.model_type = 'bart'
    elif config.architectures[0] == 'GPT2LMHeadModel':
        args.model_type = 'gpt2'
    else:
        raise ValueError('Model should be either GPT2 or BART')

    if args.prompt_column is not None and args.copy is not None and args.copy != 0:
        raise ValueError('Cannot copy from the input and use prompt at the same time. Disable either --copy or --prompt_column.')
    hyperparameters = ['temperature', 'top_k', 'top_p', 'repetition_penalty', 'num_beams']
    max_hyperparameter_len = max([len(getattr(args, h)) for h in hyperparameters])
    valid_len = [1, max_hyperparameter_len]
    for h in hyperparameters:
        if (len(getattr(args, h)) not in valid_len):
            logger.error('Hyperparameters should either have the same number of values as others or have exactly one value.')
        # If only one value is provided, use the same value for all samples
        setattr(args, h, getattr(args, h) * (max_hyperparameter_len // len(getattr(args, h))))

    logger.info('Will output %d sequences for each input.', args.batch_size*max_hyperparameter_len*args.num_samples)
    logger.info('Effective batch size for each GPU is %d', args.batch_size*args.num_samples)

    if args.gold_column is None:
        args.gold_column = args.input_column
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)
    args.model_type = args.model_type.lower()

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
        combine_files_on_disk(args.output_file, args.n_gpu, delete=True)

    else:
        run_generation(args)


def run_generation(args):
    model_class, tokenizer_class, special_tokens = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if pad_token_id is None:
        logger.error('Your tokenizer does not have a padding token')

    if args.model_type == 'gpt2':
        model.set_token_ids(end_token_id=tokenizer.convert_tokens_to_ids(special_tokens['end_token']), 
                            sep_token_id=tokenizer.convert_tokens_to_ids(special_tokens['sep_token']), 
                            pad_token_id=pad_token_id)

    logger.info(args)

    all_input_sequences, all_input_sequence_lengths, all_context_tokens, estimated_output_lengths, all_golds, reverse_maps = \
                                  create_features_from_tsv_file(file_path=args.input_file, tokenizer=tokenizer,
                                  input_column=args.input_column, gold_column=args.gold_column, prompt_column=args.prompt_column,
                                  copy=args.copy,
                                  thingtalk_column=args.thingtalk_column,
                                  sep_token=special_tokens['sep_token'], skip_heuristics=args.skip_heuristics, is_cased=args.is_cased,
                                  model_type=args.model_type)

    
    # sort contexts based on their context length so that less generated tokens are thrown away and generation can be done faster
    estimated_output_lengths, all_input_sequence_lengths, all_input_sequences, all_context_tokens, original_order, reverse_maps = \
        tuple(zip(*sorted(list(zip(estimated_output_lengths, all_input_sequence_lengths, all_input_sequences, all_context_tokens, range(len(all_context_tokens)), reverse_maps)), reverse=True)))
    all_outputs = []

    stop_token_ids = [tokenizer.convert_tokens_to_ids(stop_token) for stop_token in args.stop_tokens]
    end_token_id = tokenizer.convert_tokens_to_ids(special_tokens['end_token'])

    for batch in tqdm(range(math.ceil(len(all_context_tokens) / args.batch_size)), desc="Batch"):
        batch_slice = (batch*args.batch_size, min((batch+1)*args.batch_size, len(all_context_tokens)))
        batch_size = batch_slice[1] - batch_slice[0]
        batch_input_sequences = all_input_sequences[batch_slice[0]: batch_slice[1]]
        batch_input_sequence_lengths = all_input_sequence_lengths[batch_slice[0]: batch_slice[1]]
        batch_context_tokens = all_context_tokens[batch_slice[0]: batch_slice[1]]
        batch_reverse_maps = reverse_maps[batch_slice[0]: batch_slice[1]]
        # logger.info('batch_context_tokens = %s', str(batch_context_tokens))

        if args.model_type == 'gpt2':
            batch_context_tensor = torch.tensor(model.pad_to_max_length(batch_context_tokens), dtype=torch.long, device=args.device)
            attention_mask = None
        elif args.model_type == 'bart':
            padded_batch_context_tokens = []
            max_length = max([len(s) for s in batch_context_tokens])
            for i in range(len(batch_context_tokens)):
                padded_batch_context_tokens.append(batch_context_tokens[i]+[pad_token_id]*(max_length-len(batch_context_tokens[i])))
            batch_context_tensor = torch.tensor(padded_batch_context_tokens, dtype=torch.long, device=args.device)
            attention_mask = (batch_context_tensor!=pad_token_id).to(torch.long)
        # logger.info('context text = %s', [tokenizer.decode(b, clean_up_tokenization_spaces=False, skip_special_tokens=False) for b in batch_context_tensor])
        # logger.info('batch_context_tensor = %s', str(batch_context_tensor))

        batch_outputs = [[] for _ in range(batch_size)]
        for hyperparameter_idx in range(len(args.temperature)):
            out = model.generate(input_ids=batch_context_tensor,
                                 bad_words_ids=[[tokenizer.convert_tokens_to_ids(special_tokens['sep_token'])]] if args.model_type=='gpt2' else None,
                                 attention_mask=attention_mask,
                                 min_length=args.min_output_length,
                                 max_length=batch_context_tensor.shape[1]+args.length,
                                 num_beams=args.num_beams[hyperparameter_idx],
                                 top_k=args.top_k[hyperparameter_idx],
                                 top_p=args.top_p[hyperparameter_idx],
                                 early_stopping=True,
                                 num_return_sequences=args.num_samples,
                                 repetition_penalty=args.repetition_penalty[hyperparameter_idx],
                                 do_sample=args.temperature[hyperparameter_idx]!=0,
                                 temperature=args.temperature[hyperparameter_idx] if args.temperature[hyperparameter_idx] > 0 else 1.0, # if temperature==0, we do not sample
                                 eos_token_id=end_token_id,
                                 pad_token_id=pad_token_id
                                )
            # logger.info('out = %s', str(out))
            # logger.info('out text = %s', [tokenizer.decode(o, clean_up_tokenization_spaces=False, skip_special_tokens=False) for o in out])
            if not isinstance(out, list):
                out = out[:, :].tolist()
            for i, o in enumerate(out):
                if args.model_type=='bart':
                    o = o[1:]
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
                    text = output_heuristics(text, batch_reverse_maps[(i//args.num_samples) % batch_size])
                batch_outputs[(i//args.num_samples) % batch_size].append(text)

        all_outputs.extend(batch_outputs)


    # sort the results back to their original order
    _, all_outputs = tuple(zip(*sorted(list(zip(original_order, all_outputs)))))

    if args.output_file is not None:
        with open(args.output_file, 'w') as output_file:
            for output in all_outputs:
                for text in output:
                    output_file.write(text + '\n')
    else:
        print(json.dumps(all_outputs, indent=2))

    metrics = compute_metrics(all_outputs, all_golds, reduction=args.metric_reduction)
    logger.info('Average BLEU score = %.2f', metrics['bleu'])
    logger.info('Exact match score = %.2f', metrics['em'])

