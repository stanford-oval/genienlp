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
import torch.nn.functional as F

from transformers import GPT2Config, BartConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from .util import set_seed, get_number_of_lines, combine_files_on_disk, split_file_on_disk, get_part_path, detokenize, tokenize, lower_case, \
                    top_k_top_p_filtering, SpecialTokenMap, remove_thingtalk_quotes
from .metrics import computeBLEU
# from .models.common import BeamHypotheses


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, BartConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'bart': (BartForConditionalGeneration, BartTokenizer)
}


def apply_repetition_penalty(logits, context, repetition_penalty, prompt_token_id, pad_token_id):
    """ repetition penalty from CTRL (https://arxiv.org/abs/1909.05858), but much faster on GPU
        we penalize only the tokens that appear in the context, not in the generated text
    """
    if repetition_penalty == 1.0:
        return logits
    m = torch.scatter(input=torch.zeros_like(logits), dim=1, index=context, value=1)
    m[:prompt_token_id] = 0
    m[:pad_token_id] = 0
    # logger.info('m = ', m.shape)
    need_change = m * logits
    need_divide = need_change > 0
    need_multiply = need_change < 0
    logits = need_divide * logits / repetition_penalty + need_multiply * logits * repetition_penalty + (1-m) * logits
    
    # Old, slow implementation
    # if repetition_penalty != 1.0:
        # for i in range(context.shape[0]):
            # for _ in set(generated[i].tolist()):
                # if logits[i, _] > 0:
                    # logits[i, _] /= repetition_penalty
                # else:
                    # logits[i, _] *= repetition_penalty
    return logits


def sample_sequence(model, length, min_output_length, context, num_samples,
                    temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, device='cpu',
                    stop_token_ids=None, pad_token_id=None, supports_past=False, prompt_token_id=None, segment_token_ids=None,
                    start_reverse_position_ids=None, output_form=None):
    """
    Generates sequence of tokens for the batch of input contexts.
    Inputs:
        context: a list of token_ids, sorted by length from longest to shortest
        num_samples: the number of sequences to output for each input context
        length: The maximum length of generation in addition to the original sentence's length
        stop_token_ids: generation of each sequence will stop if we generate any of these tokens
        supports_past: set to True if the model accepts the 'past' input for more efficient generation. For example, GPT-2/Transfo-XL/XLNet/CTRL do
        segment_token_ids: a list of two integers that indicate the tokens we should use for each of the two segments
    """
    max_length = len(context[0]) # context is sorted by length from longest to shortest
    min_length = len(context[-1])

    # should not change the elements of context since it will change them outside this function as well.
    padded_context = []
    for i in range(len(context)):
        padded_context.append(context[i] + [pad_token_id] * (max_length-len(context[i]))) # pad to max_length
    
    next_index = min_length
    length = max_length + (max_length - min_length) + length # generate till max_length, then generate another max_length+length tokens
    max_index = length + next_index

    segment_ids = []
    position_ids = []
    for i in range(len(context)):
        prompt_token_position = context[i].index(prompt_token_id)
        p = list(range(prompt_token_position+1))
        segment_ids.append([segment_token_ids[0]]*len(p) + [segment_token_ids[1]]*(max_index - len(p)))
        if start_reverse_position_ids is None:
            position_ids.append(p + list(range(max_index - len(p))))
        else:
            position_ids.append(p + list(reversed(range(start_reverse_position_ids+len(p)))) + [0]*(max_index-start_reverse_position_ids-2*len(p)))

    position_ids = torch.tensor(position_ids, dtype=torch.long, device=device)
    position_ids = position_ids.repeat(num_samples, 1)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)
    segment_ids = segment_ids.repeat(num_samples, 1)

    # logger.info('context = ', context)
    # logger.info('position_ids = ', position_ids)
    # logger.info('segment_ids = ', segment_ids)

    context = torch.tensor(padded_context, dtype=torch.long, device=device)
    context = context.repeat(num_samples, 1)
    generated = context[:, :next_index]
    generated_length = torch.zeros((context.shape[0], 1), dtype=torch.long, device=device)
    should_finish = None
    generated_logits = None
    past = None
    next_token = None
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated, 'position_ids': position_ids[:, :next_index], 'token_type_ids': segment_ids[:, :next_index]}
            if supports_past:
                inputs['past'] = past
                if past is not None:
                    inputs['input_ids'] = next_token
                    inputs['position_ids'] = position_ids[:, next_index-1]
                    inputs['token_type_ids'] = segment_ids[:, next_index-1]
            
            outputs = model(**inputs)
            original_next_token_logits = outputs[0][:, -1, :]
            next_token_logits = original_next_token_logits / (temperature if temperature > 0 else 1.)
            past = outputs[1]

            next_token_logits = apply_repetition_penalty(next_token_logits, context, repetition_penalty,
                                                         prompt_token_id=prompt_token_id, pad_token_id=pad_token_id)

            if next_index < context.shape[1]:
                m = (context[:, next_index:next_index+1] != pad_token_id).long() # m==0 is where next_token should be kept
            else:
                m = torch.zeros(1, device=device)

            # prevent stop_tokens if generated_length < min_output_length
            should_remove_stop_tokens = (generated_length < min_output_length)
            next_token_logits[:, stop_token_ids] = next_token_logits[:, stop_token_ids].masked_fill(should_remove_stop_tokens, -float('Inf'))
            # logger.info('after ', next_token_logits[:, stop_token_ids])
            generated_length = generated_length + (1-m)

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            
            if output_form == 'logprob':
                generated_token_logit = F.log_softmax(original_next_token_logits, dim=-1).gather(1, next_token)
            else:
                assert output_form == 'logit'
                generated_token_logit = original_next_token_logits.gather(1, next_token)

            # throw away the tokens that we already have from the context
            if next_index < context.shape[1]:
                next_token = m*context[:, next_index:next_index+1] + (1-m)*next_token
            generated_token_logit = (1-m)*generated_token_logit

            for stop_token_id in stop_token_ids:
                if should_finish is None:
                    should_finish = ((next_token == stop_token_id) & (1-m).bool())
                else:
                    should_finish = should_finish | ((next_token == stop_token_id) & (1-m).bool())
            next_index += 1
            generated = torch.cat((generated, next_token), dim=1)
            if generated_logits is None:
                generated_logits = generated_token_logit
            else:
                generated_logits = torch.cat((generated_logits, generated_token_logit), dim=1)
            if should_finish.all():
                break
    return generated, generated_logits


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

def create_features_from_tsv_file(file_path, tokenizer, input_column, gold_column, prompt_column, copy, thingtalk_column, prompt_token,
                                  skip_heuristics, is_cased):
    """
    Read a tsv file (this includes a text file with one example per line) and returns input features that the model needs
    Outputs:

    """
    all_input_sequences = []
    all_input_sequence_lengths = []
    all_context_tokens = []
    all_context_lengths = []
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
        row = line.split('\t')
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
        input_sequence += prompt_token
        prompt = '' # includes the first few tokens of the output
        if prompt_column is not None and len(row) > prompt_column:
            prompt = row[prompt_column]
            if not skip_heuristics:
                prompt, _ = input_heuristics(prompt, thingtalk, is_cased)
                # logger.info('prompt = %s', prompt)
        input_sequence_tokens = tokenizer.encode(input_sequence, add_special_tokens=False)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        context_tokens = input_sequence_tokens + prompt_tokens
        if copy > 0:
            context_tokens.extend(context_tokens[0 : min(copy, len(context_tokens)-1)]) # -1 since we should not copy prompt_token
        all_input_sequences.append(input_sequence)
        all_input_sequence_lengths.append(len(input_sequence_tokens))
        all_context_tokens.append(context_tokens)
        all_context_lengths.append(len(context_tokens))
    
    if file_path is not None:
        input_file.close()

    return all_input_sequences, all_input_sequence_lengths, all_context_tokens, all_context_lengths, all_golds, reverse_maps


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
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
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
    
    # These can be used for improving the quality of the output
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--selection_criterion", type=str, choices=['none', 'average_logit', 'average_logprob', 'bleu'], default='none',
                        help='Select one of --num_sample outputs that maximizes this criterion')

    # These are generation hyperparameters. Each one can be a list of values in which case, we generate num_samples outputs for each set of hyperparameters.
    parser.add_argument("--start_reverse_position_ids", type=int, nargs='+', default=[None],
                        help='If provided, position ids will be the number of tokens left in generation and will start from len(input) + args.start_reverse_position_ids')
    parser.add_argument("--temperature", type=float, nargs='+', default=[1.0],
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, nargs='+', default=[1.0],
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[0.9], help='1.0 disables top-p filtering')

    parser.add_argument("--copy", type=int, default=0,
                        help='Number of tokens that will be copied at the beginning of generation. Helps preserve the original meaning of the input sequence.')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--prompt_token', type=str, default='<paraphrase>',
                        help="Token after which text generation starts. We add this to the end of all inputs.")
    parser.add_argument('--stop_tokens', type=str, nargs='+', default=['</paraphrase>'],
                        help="Token at which text generation is stopped. The first element of the list is used as segment id as well.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for text generation for each GPU.")

def main(args):
    if args.prompt_column is not None and args.copy is not None and args.copy != 0:
        raise ValueError('Cannot copy from the input and use prompt at the same time. Disable either --copy or --prompt_column.')
    hyperparameters = ['temperature', 'top_k', 'top_p', 'repetition_penalty', 'start_reverse_position_ids']
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
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)


    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    prompt_token_id = tokenizer.convert_tokens_to_ids(args.prompt_token)
    if pad_token_id is None:
        logger.error('Your tokenizer does not have a padding token')

    all_input_sequences, all_input_sequence_lengths, all_context_tokens, all_context_lengths, all_golds, reverse_maps = \
                                  create_features_from_tsv_file(file_path=args.input_file, tokenizer=tokenizer,
                                  input_column=args.input_column, gold_column=args.gold_column, prompt_column=args.prompt_column,
                                  copy=args.copy,
                                  thingtalk_column=args.thingtalk_column,
                                  prompt_token=args.prompt_token, skip_heuristics=args.skip_heuristics, is_cased=args.is_cased)

    
    # sort contexts based on their context length so that less generated tokens are thrown away and generation can be done faster
    all_context_lengths, all_input_sequence_lengths, all_input_sequences, all_context_tokens, original_order, reverse_maps = \
        tuple(zip(*sorted(list(zip(all_context_lengths, all_input_sequence_lengths, all_input_sequences, all_context_tokens, range(len(all_context_tokens)), reverse_maps)), reverse=True)))
    all_outputs = []

    stop_token_ids = [tokenizer.convert_tokens_to_ids(stop_token) for stop_token in args.stop_tokens]

    for batch in trange(math.ceil(len(all_context_tokens) / args.batch_size), desc="Batch"):
        batch_slice = (batch*args.batch_size, min((batch+1)*args.batch_size, len(all_context_tokens)))
        batch_size = batch_slice[1] - batch_slice[0]
        batch_input_sequences = all_input_sequences[batch_slice[0]: batch_slice[1]]
        batch_input_sequence_lengths = all_input_sequence_lengths[batch_slice[0]: batch_slice[1]]
        batch_context_tokens = all_context_tokens[batch_slice[0]: batch_slice[1]]
        batch_reverse_maps = reverse_maps[batch_slice[0]: batch_slice[1]]

        batch_outputs = [[] for _ in range(batch_size)]
        batch_criterion = [[] for _ in range(batch_size)]
        for hyperparameter_idx in range(len(args.temperature)):
            out, out_logits = sample_sequence(
                model=model,
                context=batch_context_tokens,
                num_samples=args.num_samples,
                length=args.length,
                min_output_length=args.min_output_length,
                temperature=args.temperature[hyperparameter_idx],
                top_k=args.top_k[hyperparameter_idx],
                top_p=args.top_p[hyperparameter_idx],
                repetition_penalty=args.repetition_penalty[hyperparameter_idx],
                device=args.device,
                stop_token_ids=stop_token_ids,
                pad_token_id=pad_token_id,
                supports_past=args.model_type in ['gpt2'],
                prompt_token_id=prompt_token_id,
                segment_token_ids=[tokenizer.convert_tokens_to_ids(args.prompt_token), tokenizer.convert_tokens_to_ids(args.stop_tokens[0])] if args.model_type=='gpt2' else [0, 1],
                start_reverse_position_ids=args.start_reverse_position_ids[hyperparameter_idx],
                output_form='logit' if args.selection_criterion=='average_logit' else 'logprob'
            )
            
            out = out[:, :].tolist()
            out_logits = out_logits[:, :].tolist()
            for i, o in enumerate(out):
                o_logits = out_logits[i]
                # logger.info('all output tokens: %s', o)
                # logger.info('all output tokens detokenized: %s', str(tokenizer.decode(o, clean_up_tokenization_spaces=True, skip_special_tokens=False)))
                o = o[batch_input_sequence_lengths[i % batch_size]:]
                # logger.info('original context tokens: %s', str(batch_context_tokens[i % batch_size]))
                # logger.info('original input sequence: %s', str(batch_input_sequences[i % batch_size]))

                if args.stop_tokens is not None:
                    min_index = len(o)
                    for stop_token_id in stop_token_ids:
                        try:
                            index = o.index(stop_token_id)
                            min_index = min(index, min_index)
                        except ValueError:
                            pass
                    if min_index < len(o) and o[min_index] == tokenizer.convert_tokens_to_ids('?'):
                        # always include the question mark
                        min_index = min_index + 1
                    if min_index < len(o) and o[min_index] == tokenizer.convert_tokens_to_ids(args.stop_tokens[0]):
                        # include </paraphrase> in logit calculation
                        o_logits = o_logits[:len(o_logits)-(len(o)-min_index-1)]
                    o = o[:min_index]
                
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True, skip_special_tokens=False)

                # assert tokenizer.pad_token not in text
                text = text.replace(tokenizer.pad_token, '')
                text = re.sub('\s\s+', ' ', text) # remove duplicate white spaces
                text = text.strip()
                if not args.skip_heuristics:
                    text = output_heuristics(text, batch_reverse_maps[i % batch_size])
                batch_outputs[i % batch_size].append(text)

                if args.selection_criterion == 'bleu':
                    # computeBLEU always converts to lower case first, so do not worry about lower/upper case here
                    criterion = computeBLEU([text], [[batch_input_sequences[i % batch_size]]])
                else:
                    criterion = np.mean(o_logits)
                batch_criterion[i % batch_size].append(criterion)
                # logger.info('generated tokens: %s', str(o))
                # logger.info('o_logits = %s', str(o_logits))
                # logger.info('generated cirterion: %.2f', criterion)
                # logger.info('text = %s', text)
                # logger.info('-'*10)


        if args.selection_criterion == 'none':
            all_outputs.extend(batch_outputs)
        else:
            for idx, example in enumerate(batch_outputs):
                logger.info('input sequence: %s', str(batch_input_sequences[idx % batch_size]))
                c, example = tuple(zip(*sorted(list(zip(batch_criterion[idx], example)), reverse=True)))
                logger.info(example)
                logger.info(c)
                logger.info('-'*10)
                selection = example[0]
                all_outputs.append([selection])

    # sort the results back to their original order
    _, all_outputs = tuple(zip(*sorted(list(zip(original_order, all_outputs)))))
    
    metrics = compute_metrics(all_outputs, all_golds, reduction=args.metric_reduction)

    if args.output_file is not None:
        with open(args.output_file, 'w') as output_file:
            if args.output_file is not None:
                for _ in all_outputs:
                    for text in _:
                        output_file.write(text + '\n')
    else:
        print(json.dumps(all_outputs, indent=2))
    logger.info('Average BLEU score = %.2f', metrics['bleu'])
    logger.info('Exact match score = %.2f', metrics['em'])

