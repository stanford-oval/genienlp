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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from tqdm import trange, tqdm
import math
import json
import csv
import re
import copy

# multiprocessing with CUDA
from torch.multiprocessing import Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import torch
import torch.nn.functional as F

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer

from .util import set_seed, get_number_of_lines, combine_files_on_disk, split_file_on_disk, get_file_part_path, detokenize, top_k_top_p_filtering


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def sample_sequence(model, length, context, position_ids, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu',
                    stop_token_ids=None, pad_token_id=None, supports_past=False, prompt_token_id=None, segment_token_ids=None):
    """
    Generates sequence of tokens for the batch of input contexts.
    Inputs:
        context: a list of token_ids, sorted by length from longest to shortest
        position_ids: a list of indicate that indicates the positional embedding we should use for each token in context
        num_samples: the number of sequences to output for each input context
        length: The maximum length of generation in addition to the original sentence's length
        stop_token_ids: generation of each sequence will stop if we generate any of these tokens
        supports_past: set to True if the model accepts the 'past' input for more efficient generation. For example, GPT-2/Transfo-XL/XLNet/CTRL do
        segment_token_ids: a list of two integers that indicate the tokens we should use for each of the two segments
    """
    max_length = len(context[0]) # context is sorted by length from longest to shortest
    min_length = len(context[-1])
    for a in context:
        a.extend([pad_token_id] * (max_length-len(a)))
    
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.repeat(num_samples, 1)
    next_index = min_length
    generated = context[:, :next_index]
    should_finish = None
    length = max_length + length
    segment_ids = []
    for p in position_ids:
        segment_ids.append([segment_token_ids[0]]*len(p)+[segment_token_ids[1]]*(length+max_length-len(p)))
        p.extend(range(length+max_length-len(p)))

    position_ids = torch.tensor(position_ids, dtype=torch.long, device=device)
    position_ids = position_ids.repeat(num_samples, 1)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)
    segment_ids = segment_ids.repeat(num_samples, 1)

    # print('context = ', context)
    # print('position_ids = ', position_ids)
    # print('segment_ids = ', segment_ids)

    past = None
    next_token = None
    with torch.no_grad():
        # rep_penalty = np.random.random(length) < 0.1
        # original_rep_penalty = repetition_penalty
        # print('rep_penalty = ', rep_penalty)
        for _ in trange(length):
            inputs = {'input_ids': generated, 'position_ids': position_ids[:, :next_index], 'token_type_ids': segment_ids[:, :next_index]}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            if is_xlm_mlm and xlm_mask_token:
                # XLM MLM models are direct models (predict same token, not next token)
                # => need one additional dummy token in the input (will be masked and guessed)
                input_ids = torch.cat((generated, torch.full((1, 1), xlm_mask_token, dtype=torch.long, device=device)), dim=1)
                inputs = {'input_ids': input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            if supports_past:
                inputs['past'] = past
                if past is not None:
                    inputs['input_ids'] = next_token
                    inputs['position_ids'] = position_ids[:, next_index-1]
                    inputs['token_type_ids'] = segment_ids[:, next_index-1]
            
            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            past = outputs[1]

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858), but much faster on GPU
            # for repetition_penalty, we penalize the tokens that appear in the context
            m = torch.scatter(input=torch.zeros_like(next_token_logits), dim=1, index=context, value=1)
            m[:prompt_token_id] = 0
            m[:pad_token_id] = 0
            # print('m = ', m.shape)
            need_change = m * next_token_logits
            need_divide = need_change > 0
            need_multiply = need_change < 0
            next_token_logits = need_divide * next_token_logits / repetition_penalty + need_multiply * next_token_logits * repetition_penalty + (1-m) * next_token_logits
            
            # Old, slow implementation
            # if repetition_penalty != 1.0:
                # for i in range(context.shape[0]):
                    # for _ in set(generated[i].tolist()):
                        # if next_token_logits[i, _] > 0:
                            # next_token_logits[i, _] /= repetition_penalty
                        # else:
                            # next_token_logits[i, _] *= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)


            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # throw away the tokens that we already have from the context
            if next_index < context.shape[1]:
                m = (context[:, next_index:next_index+1] != pad_token_id).long()
                next_token = m*context[:, next_index:next_index+1]+(1-m)*next_token
            else:
                m = torch.zeros(1, device=device)

            for stop_token_id in stop_token_ids:
                if should_finish is None:
                    should_finish = ((next_token == stop_token_id) & (1-m).bool())
                else:
                    should_finish = should_finish | ((next_token == stop_token_id) & (1-m).bool())
            next_index += 1
            generated = torch.cat((generated, next_token), dim=1)
            if should_finish.all():
                break
    return generated


special_token_mapping = {
    'PATH_NAME_0': {'forward': 'my1folder'},
    'PATH_NAME_1': {'forward': 'my2folder'},
    'TIME_0': {'forward': '1p.m.', 'back': ['1 pm', '1pm', '1:00 pm', '1:00pm', '1p.m.', '1 p.m.', '1:00 p.m.', '1:00']},
    'TIME_1': {'forward': '2p.m.', 'back': ['2 pm', '2pm', '2:00 pm', '2:00pm', '2p.m.', '2 p.m.', '2:00 p.m.', '2:00']},
    'EMAIL_ADDRESS_0': {'forward': 'e1@example.com'},
    'EMAIL_ADDRESS_1': {'forward': 'e2@example.com'},
    'URL_0': {'forward': 'my1site.com'},
    'URL_1': {'forward': 'my2site.com'},
    'DATE_0': {'forward': '5-6-2015', 'back': ['5-6-2015']},
    'DATE_1': {'forward': '8-3-2016', 'back': ['8-3-2016']},
    'CURRENCY_0': {'forward': '$12', 'back': ['$12', 'twelve dollars', '12 dollars', '$ 12', '$ 12.00', '12.00', '12']},
    'CURRENCY_1': {'forward': '$13', 'back': ['$13', 'thirteen dollars', '13 dollars', '$ 13', '$ 13.00', '13.00', '13']},
    'NUMBER_0': {'forward': '2', 'back': ['2', 'two']},
    'NUMBER_1': {'forward': '3', 'back': ['3', 'three']},
    'DURATION_0': {'forward': '5 weeks', 'back': ['5 weeks', 'five weeks']},
    'DURATION_1': {'forward': '6 weeks', 'back': ['6 weeks', 'six weeks']},
    'LOCATION_0': {'forward': 'locatio1n', 'back': ['locatio1n', 'locat1n']},
    'LOCATION_1': {'forward': 'locatio2n', 'back': ['locatio2n', 'locat2n']},
    'PHONE_NUMBER_0': {'forward': '888-8888'},
    'PHONE_NUMBER_1': {'forward': '777-8888'}
}

def input_heuristics(s: str):
    """
    Changes the input string so that it is closer to what the pre-traied language models have seen during their training.
    Outputs:
        s: the new string
        reverse_map: a list of special tokens. Can be used to recover the original special_tokens in the string
    """
    reverse_map = []
    s = s.strip()
    s = detokenize(s)

    # Put question mark at the end whenever necessary.
    if s.startswith('which') or s.startswith('what') or s.startswith('where') or s.startswith('how') or s.startswith('who') or s.startswith('when'):
        if s.endswith('.'):
            s = s[:-1]
        s += '?'

    # replace special tokens with natural-looking exmaples
    for special_token, natural_form in special_token_mapping.items():
        new_s = s.replace(special_token, natural_form['forward'])
        if new_s != s:
            print(new_s)
            reverse_map.append(special_token)
        s = new_s
    return s, reverse_map

def output_heuristics(s: str, reverse_map: list):
    s = s.replace('<pad>', '')
    s = re.sub('\s\s+', ' ', s) # remove multiple white spaces
    s = s.strip()

    for special_token in reverse_map:
        if 'back' in special_token_mapping[special_token]:
            back = special_token_mapping[special_token]['back']
        else:
            back = [special_token_mapping[special_token]['forward']]
        for b in back:
            if b in s:
                s = s.replace(b, special_token)
                break
    return s

def parse_argv(parser):
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--input_file", type=str, help="The file from which we read prompts.")
    parser.add_argument('--input_column', type=int, required=True,
                        help='The column in the input file which contains the input sentences.')
    parser.add_argument("--output_file", type=str, help="When specified, generated text will be written in this file.")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=10, help='The generated sentences will have a maximum length of len(input) + arg.length')
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--prompt_token', type=str, default='<paraphrase>',
                        help="Token after which text generation starts. We add this to the end of all inputs.")
    parser.add_argument('--stop_tokens', type=str, nargs='+', default=['</paraphrase>', '?'],
                        help="Token at which text generation is stopped. The first element of the list is used as segment id as well.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for text generation for each GPU.")

def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)
    args.model_type = args.model_type.lower()

    if args.n_gpu > 1:
        # Independent multi-GPU evaluation
        all_processes = []
        all_input_files = split_file_on_disk(args.input_file, args.n_gpu)
        for gpu_idx in range(args.n_gpu):
            copy_args = copy.copy(args)
            if torch.cuda.is_available() and not args.no_cuda:
                copy_args.device = torch.device("cuda:" + str(gpu_idx))
            copy_args.n_gpu = 1
            copy_args.input_file = all_input_files[gpu_idx]
            copy_args.output_file = get_file_part_path(args.output_file, gpu_idx)
            
            p = Process(target=run_generation, args=(copy_args,))
            all_processes.append(p)
            p.start()

        for p in all_processes:
            p.join()

        combine_files_on_disk(args.output_file, args.n_gpu)

    else:
        run_generation(args)



def run_generation(args):
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    model.eval()
    print(args.stop_tokens)

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)
    if args.model_type in ["ctrl"]:
        if args.temperature > 0.7:
            logger.info('CTRL typically works better with lower temperatures (and lower top_k).')

    xlm_lang = None
    # XLM Language usage detailed in the issues #1414
    if args.model_type in ["xlm"] and hasattr(tokenizer, 'lang2id') and hasattr(model.config, 'use_lang_emb') \
            and model.config.use_lang_emb:
        if args.xlm_lang:
            language = args.xlm_lang
        else:
            language = None
            while language not in tokenizer.lang2id.keys():
                language = input("Using XLM. Select language in " + str(list(tokenizer.lang2id.keys())) + " >>> ")
        xlm_lang = tokenizer.lang2id[language]

    # XLM masked-language modeling (MLM) models need masked token (see details in sample_sequence)
    is_xlm_mlm = args.model_type in ["xlm"] and 'mlm' in args.model_name_or_path
    if is_xlm_mlm:
        xlm_mask_token = tokenizer.mask_token_id
    else:
        xlm_mask_token = None

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    prompt_token_id = tokenizer.convert_tokens_to_ids(args.prompt_token)
    if pad_token_id is None:
        logger.error('Your tokenizer does not have a padding token')

    all_context_tokens = []
    all_context_lengths = []
    all_position_ids = []
    reverse_maps = []
    number_of_lines = get_number_of_lines(args.input_file)
    with open(args.input_file) as input_file:
        reader = csv.reader(input_file, delimiter='\t')
        for row in tqdm(reader, desc='Reading Input File', total=number_of_lines):
            raw_text = row[args.input_column]
            # print('before text = ', raw_text)
            raw_text, reverse_map = input_heuristics(raw_text)
            reverse_maps.append(reverse_map)
            # print('after text = ', raw_text)
            raw_text += args.prompt_token
            if args.model_type in ["transfo-xl", "xlnet"]:
                # Models with memory likes to have a long prompt for short inputs.
                raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
            context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
            position_ids = [pos for pos in range(len(context_tokens))]
            all_context_tokens.append(context_tokens)
            all_context_lengths.append(len(context_tokens))
            all_position_ids.append(position_ids)
            if args.model_type == "ctrl":
                if not any(context_tokens[0] == x for x in tokenizer.control_codes.values()):
                    logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    
    # sort contexts based on their length so that less generated tokens are thrown away and generation can be done faster
    t = list(zip(*sorted(list(zip(all_context_lengths, all_context_tokens, all_position_ids, range(len(all_context_tokens)), reverse_maps)), reverse=True)))
    all_context_lengths, all_context_tokens, all_position_ids, original_order, reverse_maps = list(t[0]), list(t[1]), list(t[2]), list(t[3]), list(t[4])
    all_outputs = []

    if args.output_file is not None:
        output_file = open(args.output_file, 'w')


    for batch in trange(math.ceil(len(all_context_tokens) / args.batch_size), desc="Batch"):
        batch_slice = (batch*args.batch_size, min((batch+1)*args.batch_size, len(all_context_tokens)))
        batch_context_tokens = all_context_tokens[batch_slice[0]: batch_slice[1]]
        batch_context_lengths = all_context_lengths[batch_slice[0]: batch_slice[1]]
        batch_position_ids = all_position_ids[batch_slice[0]: batch_slice[1]]
        batch_reverse_maps = reverse_maps[batch_slice[0]: batch_slice[1]]

        out = sample_sequence(
            model=model,
            context=batch_context_tokens,
            position_ids=batch_position_ids,
            num_samples=args.num_samples,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            is_xlnet=bool(args.model_type == "xlnet"),
            is_xlm_mlm=is_xlm_mlm,
            xlm_mask_token=xlm_mask_token,
            xlm_lang=xlm_lang,
            device=args.device,
            stop_token_ids=[tokenizer.convert_tokens_to_ids(stop_token) for stop_token in args.stop_tokens],
            pad_token_id=pad_token_id,
            supports_past=args.model_type in ['gpt2', 'openai-gpt', 'transfo-xl', 'xlnet', 'ctrl'],
            prompt_token_id=prompt_token_id,
            segment_token_ids=[tokenizer.convert_tokens_to_ids(args.prompt_token), tokenizer.convert_tokens_to_ids(args.stop_tokens[0])]
        )
        out = out[:, :].tolist()
        batch_outputs = [[] for _ in range(batch_slice[1]-batch_slice[0])]
        for i, o in enumerate(out):
            o = o[batch_context_lengths[i % (batch_slice[1]-batch_slice[0])]:]
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True, skip_special_tokens=False)
            # print('original tokens: ', batch_context_tokens[i % (batch_slice[1]-batch_slice[0])])
            # print('generated tokens: ', o)
            # print('original text: ', tokenizer.decode(batch_context_tokens[i % (batch_slice[1]-batch_slice[0])], clean_up_tokenization_spaces=True, skip_special_tokens=False))
            # print('text = ', text)
            if args.stop_tokens is not None:
                min_index = len(text)
                for stop_token in args.stop_tokens:
                    index = text.find(stop_token)
                    if index >= 0:
                        min_index = min(index, min_index)
                if min_index < len(text) and text[min_index] == '?':
                    min_index += 1
                text = text[:min_index]

            text = output_heuristics(text, batch_reverse_maps[i % (batch_slice[1]-batch_slice[0])])
            batch_outputs[i % (batch_slice[1]-batch_slice[0])].append(text)

        all_outputs.extend(batch_outputs)

    # sort the results back to their original order
    t = list(zip(*sorted(list(zip(original_order, all_outputs)))))
    all_outputs = list(t[1])

    if args.output_file is not None:
        for _ in all_outputs:
            for text in _:
                output_file.write(text + '\n')
    else:
        print(json.dumps(all_outputs, indent=2))

