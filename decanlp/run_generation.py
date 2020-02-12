#!/usr/bin/env python3
# coding=utf-8
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

import argparse
import logging
from tqdm import trange
import math
import json
import csv
import sys

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer


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


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu',
                    stop_token_id=None, pad_token_id=None, supports_past=False):
    """
    Generates sequence of tokens for the batch of input contexts.
    Inputs:
        context: a list of token_ids
        num_samples: the number of sequences to output for each input context
        length: The maximum length of generation
        stop_token_id: generation of each sequence will stop if we generate this token
        supports_past: set to True if the model accepts the 'past' input for more efficient generation. For example, GPT-2/Transfo-XL/XLNet/CTRL do
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
    # print('generated = ', generated)
    length = max(length+max_length-min_length, max_length)
    past = None
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
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
            # print('input_ids = ', inputs['input_ids'])
            outputs = model(**inputs)
            # print('outputs[0] = ', outputs[0].shape)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            # print('next_token_logits = ', next_token_logits)
            past = outputs[1]
            # print('len(past) = ', len(past))
            # print('past[0] = ', past[0].shape)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(context.shape[0]):
                    for _ in set(generated[i].tolist()):
                        if next_token_logits[i, _] > 0:
                            next_token_logits[i, _] /= repetition_penalty
                        else:
                            next_token_logits[i, _] *= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # throw away the tokens that we already have from the context
            if next_index < context.shape[1]:
                # print('context[:, next_index:next_index+1] = ', context[:, next_index:next_index+1])
                m = (context[:, next_index:next_index+1] != pad_token_id).long()
                # print('m = ', m)
                next_token = m*context[:, next_index:next_index+1]+(1-m)*next_token
                # print('next_token = ', next_token)

            if should_finish is None:
                should_finish = (next_token == stop_token_id)
            else:
                should_finish = should_finish | (next_token == stop_token_id)
            next_index += 1
            generated = torch.cat((generated, next_token), dim=1)
            if should_finish.all():
                break
    return generated


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(prog=argv[0])
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
    parser.add_argument("--length", type=int, default=20)
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
                        help="Token after which text generation starts. We need to add this to the end of all inputs.")
    parser.add_argument('--stop_token', type=str, default='</paraphrase>',
                        help="Token at which text generation is stopped")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for text generation.")
    args = parser.parse_args(argv[1:])

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
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

    all_context_tokens = []
    all_context_lengths = []
    with open(args.input_file) as input_file:
        reader = csv.reader(input_file, delimiter='\t')
        for row in reader:
            raw_text = row[args.input_column]
            raw_text += args.prompt_token
            if args.model_type in ["transfo-xl", "xlnet"]:
                # Models with memory likes to have a long prompt for short inputs.
                raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
            context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
            all_context_tokens.append(context_tokens)
            all_context_lengths.append(len(context_tokens))
            if args.model_type == "ctrl":
                if not any(context_tokens[0] == x for x in tokenizer.control_codes.values()):
                    logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if pad_token_id is None:
        logger.error('Your tokenizer does not have a padding token')

    # sort contexts based on their length so that less generated tokens are thrown away and generation can be done faster
    t = list(zip(*sorted(list(zip(all_context_lengths, all_context_tokens, range(len(all_context_tokens)))), reverse=True)))
    all_context_lengths, all_context_tokens, original_order = list(t[0]), list(t[1]), list(t[2])
    all_outputs = []

    if args.output_file is not None:
        output_file = open(args.output_file, 'w')

    for batch in trange(math.ceil(len(all_context_tokens) / args.batch_size), desc="Batch"):
        batch_slice = (batch*args.batch_size, min((batch+1)*args.batch_size, len(all_context_tokens)))
        batch_context_tokens = all_context_tokens[batch_slice[0]: batch_slice[1]]
        batch_context_lengths = all_context_lengths[batch_slice[0]: batch_slice[1]]
        # print('batch_slice = ', batch_slice)
        # print('all_context_lengths = ', all_context_lengths)
        # print('batch_context_lengths = ', batch_context_lengths)

        out = sample_sequence(
            model=model,
            context=batch_context_tokens,
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
            stop_token_id=tokenizer.convert_tokens_to_ids(args.stop_token),
            pad_token_id=pad_token_id,
            supports_past=args.model_type in ['gpt2', 'openai-gpt', 'transfo-xl', 'xlnet', 'ctrl']
        )
        out = out[:, :].tolist()
        # print('pad = ', pad_token_id)
        # print('stop token = ', tokenizer.convert_tokens_to_ids(args.stop_token))
        batch_outputs = [[] for _ in range(batch_slice[1]-batch_slice[0])]
        for i, o in enumerate(out):
            # print('len(o) = ', len(o))
            # print('o = ', o)
            # print(i % (batch_slice[1]-batch_slice[0]))
            # print('context_length = ', batch_context_lengths[i % (batch_slice[1]-batch_slice[0])])
            o = o[batch_context_lengths[i % (batch_slice[1]-batch_slice[0])]:]
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True, skip_special_tokens=False)

            # print('len(o) = ', len(o))
            # print('o = ', o)
            if args.stop_token:
                index = text.find(args.stop_token)
                if index == -1:
                    index = None
                text = text[:index]
            text = text.strip()
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


if __name__ == '__main__':
    main()
