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
"""
This script in used for text generation using library models.
It currently supports paraphrasing and translation tasks.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import re

import numpy as np
from torch.multiprocessing import Process, set_start_method

from ..data_utils.almond_utils import tokenize_cjk_chars
from ..data_utils.progbar import prange
from ..model_utils.translation import compute_attention
from .data_utils import create_features_from_tsv_file, output_heuristics
from .model_utils import compute_metrics, force_replace_quoted_params, replace_quoted_params

try:
    set_start_method('spawn')
except RuntimeError:
    pass

import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    GPT2Tokenizer,
    MarianMTModel,
    MarianTokenizer,
    MBart50Tokenizer,
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
    PretrainedConfig,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from ..model_utils.transformers_utils import GenieMBartTokenizer
from ..util import combine_files_on_disk, get_part_path, set_seed, split_file_on_disk
from .data_utils import group_together
from .GPT2Seq2Seq import GPT2Seq2Seq
from .model_utils import check_args

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Seq2Seq, GPT2Tokenizer, {'bos_token': '<unk>', 'sep_token': '<paraphrase>', 'eos_token': '</paraphrase>'}),
    't5': (T5ForConditionalGeneration, T5Tokenizer, {'bos_token': '<unk>', 'sep_token': '<unk>', 'eos_token': '</s>'}),
    'mt5': (MT5ForConditionalGeneration, T5Tokenizer, {'bos_token': '<unk>', 'sep_token': '<unk>', 'eos_token': '</s>'}),
    'bart': (BartForConditionalGeneration, BartTokenizer, {'bos_token': '<s>', 'sep_token': '<unk>', 'eos_token': '</s>'}),
    'mbart': (
        MBartForConditionalGeneration,
        GenieMBartTokenizer,
        {'bos_token': '<s>', 'sep_token': '<unk>', 'eos_token': '</s>'},
    ),
    'mbart50': (
        MBartForConditionalGeneration,
        MBart50Tokenizer,
        {'bos_token': '<s>', 'sep_token': '<unk>', 'eos_token': '</s>'},
    ),
    'marian': (MarianMTModel, MarianTokenizer, {'bos_token': '<unk>', 'sep_token': '<unk>', 'eos_token': '</s>'}),
}


def parse_argv(parser):
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument("--input_file", type=str, help="The file from which we read prompts. Defaults to stdin.")
    parser.add_argument(
        '--input_column', type=int, required=True, help='The column in the input file which contains the input sentences.'
    )
    parser.add_argument(
        '--prompt_column',
        type=int,
        default=None,
        help='The column in the input file which contains the text we should start generation from.',
    )
    parser.add_argument(
        '--gold_column',
        type=int,
        default=None,
        help='The column in the input file which contains the gold sentences. Defaults to --input_column if no gold is available.',
    )
    parser.add_argument(
        '--thingtalk_column', type=int, default=None, help='The column in the input file which contains the ThingTalk program.'
    )
    parser.add_argument(
        '--id_column', type=int, default=None, help='The column in the input file which contains the example ID.'
    )
    parser.add_argument(
        "--output_file", type=str, help="When specified, generated text will be written in this file. Defaults to stdout."
    )
    parser.add_argument(
        "--intermediate_file", type=str, default='./paraphrase_tmp.tsv', help="Used to save intermediate results."
    )

    parser.add_argument(
        '--output_prompt',
        action='store_true',
        help='Whether we should include the prompt (specified via --prompt_column or --copy) in the output sequence',
    )
    parser.add_argument("--max_input_length", type=int, default=512, help='Crop longer sentences by this maximum length')
    parser.add_argument(
        "--length", type=int, default=15, help='The generated sentences will have a maximum length of len(input) + arg.length'
    )
    parser.add_argument(
        "--min_output_length",
        type=int,
        default=2,
        help='Will prevent stop tokens from appearing in the first --min_output_length tokens of the generated sentences.',
    )
    parser.add_argument(
        "--skip_heuristics", action='store_true', help='If True, will not replace special word such as NUMBER_0 in the input.'
    )
    parser.add_argument(
        "--is_cased",
        action='store_true',
        help='If True, the trained model is cased, so if --skip_heuristics is not set, we will convert the input to upper case and the output back to lower case.',
    )
    parser.add_argument(
        "--metric_reduction",
        type=str,
        choices=['average', 'max'],
        default='average',
        help="How we should calculate metrics where there are multiple generations per example.",
    )

    parser.add_argument(
        "--shuffle_input",
        action='store_true',
        help='If set, we will shuffle input dataset before processing it'
        'Used mainly with subsampling so we take different portion of data each time',
    )

    parser.add_argument(
        "--pipe_mode", action='store_true', help='If set, we will generate paraphrases of paraphrases of ... as well.'
    )
    # These are generation hyperparameters. Each one can be a list of values in which case, we generate num_samples outputs for each set of hyperparameters.
    parser.add_argument("--num_samples", type=int, nargs='+', default=[1])
    parser.add_argument("--temperature", type=float, nargs='+', default=[1.0], help="temperature of 0 implies greedy sampling")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        nargs='+',
        default=[1.0],
        help="primarily useful for CTRL model; in that case, use 1.2",
    )
    parser.add_argument("--top_k", type=int, nargs='+', default=[0], help='0 disables top-k filtering')
    parser.add_argument("--top_p", type=float, nargs='+', default=[0.9], help='1.0 disables top-p filtering')
    parser.add_argument("--num_beams", type=int, nargs='+', default=[1], help='1 disables beam seach')
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        nargs='+',
        default=[0],
        help='ngrams of this size cannot be repeated in the output. 0 disables it.',
    )

    parser.add_argument(
        "--copy",
        type=int,
        default=0,
        help='Number of tokens that will be copied at the beginning of generation. Helps preserve the original meaning of the input sequence.',
    )
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        '--stop_tokens',
        type=str,
        nargs='+',
        default=[],
        help="Tokens (other than the model-specific `eos_token`) at which text generation should be stopped.",
    )
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for text generation for each GPU.")

    parser.add_argument(
        '--pad_token', type=str, default='<pad>', help='The special token for padding, if tokenizer does not have that'
    )

    parser.add_argument(
        '--cache_dir',
        default='.embeddings',
        type=str,
        help='where to save transforemrs cached models, configs, and tokenizers.',
    )

    parser.add_argument(
        '--trained_model_type', type=str, help='if provided we make sure the loaded model matches the model_type'
    )

    parser.add_argument('--src_lang', type=str, help='source language used for translation task')
    parser.add_argument('--tgt_lang', type=str, help='target language used for translation task')
    parser.add_argument(
        '--output_attentions', action='store_true', help='return self and cross attention weights for seq2seq models'
    )
    parser.add_argument('--output_hidden_states', action='store_true', help='return all hidden states for seq2seq models')

    parser.add_argument(
        '--att_pooling',
        type=str,
        default='max',
        help='pooling used to calculate decoder-encoder attention values across different heads',
    )
    parser.add_argument('--plot_heatmaps', action='store_true', help='whether to plot decoder-encoder attention heatmaps')
    parser.add_argument(
        '--replace_qp', action='store_true', help='replace parameter values after translation with source values'
    )
    parser.add_argument(
        '--force_replace_qp',
        action='store_true',
        help='if we parameters could not be replaced leveraging quotation marks,'
        ' rely purely on attention to find text spans',
    )
    parser.add_argument('--subsample', type=int, default=20000000, help='subsample input datasets')
    parser.add_argument('--task', type=str, required=True, choices=['paraphrase', 'translate'])
    parser.add_argument(
        "--output_example_ids_too", action='store_true', help='Generate two column output with ids in the first column'
    )

    parser.add_argument(
        '--mask_tokens', action='store_true', help='mask input tokens and infill them using denoising pretrained model'
    )
    parser.add_argument(
        '--mask_token_prob', type=float, default=0.15, help='Probability of an input token being masked in the sentence'
    )
    parser.add_argument(
        '--delete_tokens',
        action='store_true',
        help='delete input tokens and infill them using denoising pretrained model'
        'In contrast to token masking, the model should decide which positions have missing inputs',
    )
    parser.add_argument(
        '--delete_token_prob', type=float, default=0.15, help='Probability of an input token being deleted in the sentence'
    )
    parser.add_argument(
        '--infill_text', action='store_true', help='mask consecutive tokens and infill them using denoising pretrained model'
    )
    parser.add_argument(
        '--num_text_spans', type=int, default=3, help='number of text spans to sample for text infilling method'
    )
    parser.add_argument('--infill_max_tries', type=int, default=3, help='Maximum number of tries to find an appropriate span')

    parser.add_argument(
        '--permute_sentences',
        action='store_true',
        help='divide document into sentences based on fill stops and'
        'permutate them. Use this only if input has multiple sentences.',
    )
    parser.add_argument(
        '--rotate_sentence',
        action='store_true',
        help='a pivot token is chosen randomly, and sentence is rotated so new sentence start with pivot token',
    )

    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit. On certain GPUs (e.g. Nvidia V100) improves the inference speed",
    )
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument('--verbose', action='store_true', help='log additional information for debugging purposes')

    parser.add_argument('--no_fast_tokenizer', action='store_true', help='Use slow version of huggingface tokenizer')


def main(args):
    hyperparameters = [
        'num_samples',
        'temperature',
        'top_k',
        'top_p',
        'repetition_penalty',
        'num_beams',
        'no_repeat_ngram_size',
    ]
    max_hyperparameter_len = max([len(getattr(args, h)) for h in hyperparameters])
    valid_len = [1, max_hyperparameter_len]
    for h in hyperparameters:
        if len(getattr(args, h)) not in valid_len:
            logger.error('Hyperparameters should either have the same number of values as others or have exactly one value.')
        # If only one value is provided, use the same value for all samples
        setattr(args, h, getattr(args, h) * (max_hyperparameter_len // len(getattr(args, h))))

    logger.info(
        'Will output %d sequences for each input.', sum(args.num_samples) if not args.pipe_mode else np.prod(args.num_samples)
    )
    logger.info('Effective batch size for each device is %d', args.batch_size * max(args.num_samples))

    # TODO using intermediate files for pipe_mode is not clean. It needs to change.
    if args.pipe_mode:
        intermediate_files = [args.input_file] + [args.intermediate_file + str(i) for i in range(max_hyperparameter_len)]
        for i in range(max_hyperparameter_len):
            copy_args = copy.copy(args)
            for h in hyperparameters:
                setattr(copy_args, h, [getattr(args, h)[i]])
            copy_args.input_file = intermediate_files[i]
            copy_args.output_file = intermediate_files[i + 1]
            run_multi_process_generation(copy_args)
        all_outputs = group_together(intermediate_files[1:], args.num_samples)
        for file_path in intermediate_files[1:]:
            os.remove(file_path)
        if args.output_file is not None:
            if not os.path.exists(os.path.dirname(args.output_file)):
                os.makedirs(os.path.dirname(args.output_file), exist_ok=False)
            with open(args.output_file, 'w') as output_file:
                for output in all_outputs:
                    for text in output:
                        output_file.write(text + '\n')
        else:
            print(json.dumps(all_outputs, indent=2))
    else:
        run_multi_process_generation(args)


def run_multi_process_generation(args):
    config = PretrainedConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    # get model type from saved config
    if hasattr(config, 'model_type'):
        args.model_type = getattr(config, 'model_type')
        if args.model_type == 'mbart' and '-50-' in args.model_name_or_path:
            args.model_type = 'mbart50'
    else:
        raise ValueError('Model should be either GPT2, BART, MBART, or Marian')

    # check arguments validity
    check_args(args)

    if sum([args.mask_tokens, args.delete_tokens, args.infill_text, args.permute_sentences, args.rotate_sentence]) >= 2:
        raise ValueError('Mixing denoising techniques is unlikely to work. Please use one method per run')

    if (args.mask_tokens or args.delete_tokens or args.rotate_sentence) and args.model_type == 'mbart':
        raise ValueError(
            'MBART is pretrained only with text_infilling and permute_sentences noising methods. '
            'Applying other noising techniques is unlikely to work'
        )

    if args.trained_model_type and args.trained_model_type != '' and args.model_type != args.trained_model_type:
        raise ValueError('The loaded model type does not match with what the user provided')

    if args.prompt_column is not None and args.copy is not None and args.copy != 0:
        raise ValueError(
            'Cannot copy from the input and use prompt at the same time. Disable either --copy or --prompt_column.'
        )

    if args.gold_column is None:
        args.gold_column = args.input_column
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if args.output_file is not None:
        if not os.path.exists(os.path.dirname(args.output_file)):
            os.makedirs(os.path.dirname(args.output_file), exist_ok=False)

    set_seed(args)

    if args.n_gpu > 1:
        if args.input_file is None:
            raise ValueError('Cannot use multiple GPUs when reading from stdin. You should provide an --input_file')
        logger.info('Running generation in parallel on {} GPUs'.format(args.n_gpu))
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

            p = Process(target=run_single_process_generation, args=(copy_args, config))
            all_processes.append(p)
            p.start()

        for p in all_processes:
            p.join()

        for file in all_input_files:
            os.remove(file)
        combine_files_on_disk(args.output_file, args.n_gpu, line_group_size=sum(args.num_samples), delete=True)

    else:
        run_single_process_generation(args, config)


def run_single_process_generation(args, config):
    model_class, tokenizer_class, special_tokens = MODEL_CLASSES[args.model_type]

    output_attentions = args.output_attentions
    output_hidden_states = args.output_hidden_states

    model = model_class.from_pretrained(
        args.model_name_or_path,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_dir=args.cache_dir,
    )
    model.to(args.device)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    model.eval()
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir, use_fast=not args.no_fast_tokenizer
    )
    eos_token_id = tokenizer.convert_tokens_to_ids(special_tokens['eos_token'])
    sep_token_id = tokenizer.convert_tokens_to_ids(special_tokens['sep_token'])

    if tokenizer.pad_token is None:
        # this assigns pad token but doesn't add it to the vocabulary
        tokenizer.pad_token = args.pad_token

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    if pad_token_id is None:
        logger.error('Your tokenizer does not have a padding token')

    if args.model_type == 'gpt2':
        model.set_token_ids(eos_token_id=eos_token_id, sep_token_id=sep_token_id, pad_token_id=pad_token_id)

    logger.info(args)

    model_input_prefix = ''
    if args.model_type == 'marian' and args.tgt_lang:
        # TODO check if extra space after pattern is necessary
        model_input_prefix = '>>{}<< '.format(args.tgt_lang)
    elif args.model_type == 't5':
        if args.task == 'translate':
            t5_task = 'translation_{}_to_{}'.format(args.src_lang, args.tgt_lang)
        else:
            t5_task = 'summarization'
        model_input_prefix = config.task_specific_params[t5_task]['prefix']

    masking_token = getattr(tokenizer, 'mask_token', '<mask>')

    (
        all_input_sequences,
        all_input_sequence_lengths,
        all_example_ids,
        all_context_ids,
        estimated_output_lengths,
        all_golds,
        reverse_maps,
        all_prompt_ids,
    ) = create_features_from_tsv_file(
        file_path=args.input_file,
        tokenizer=tokenizer,
        input_column=args.input_column,
        gold_column=args.gold_column,
        id_column=args.id_column,
        prompt_column=args.prompt_column,
        thingtalk_column=args.thingtalk_column,
        copy=args.copy,
        sep_token_id=sep_token_id,
        skip_heuristics=args.skip_heuristics,
        is_cased=args.is_cased,
        model_type=args.model_type,
        src_lang=args.src_lang,
        subsample=args.subsample,
        shuffle_input=args.shuffle_input,
        task=args.task,
        model_input_prefix=model_input_prefix,
        max_input_length=args.max_input_length,
        mask_tokens=args.mask_tokens,
        mask_token_prob=args.mask_token_prob,
        masking_token=masking_token,
        infill_max_tries=args.infill_max_tries,
        delete_tokens=args.delete_tokens,
        delete_token_prob=args.delete_token_prob,
        infill_text=args.infill_text,
        num_text_spans=args.num_text_spans,
        permute_sentences=args.permute_sentences,
        rotate_sentence=args.rotate_sentence,
    )

    # sort contexts based on their context length so that less generated tokens are thrown away and generation can be done faster
    (
        estimated_output_lengths,
        all_input_sequence_lengths,
        all_input_sequences,
        all_context_ids,
        original_order,
        reverse_maps,
        all_prompt_ids,
    ) = tuple(
        zip(
            *sorted(
                list(
                    zip(
                        estimated_output_lengths,
                        all_input_sequence_lengths,
                        all_input_sequences,
                        all_context_ids,
                        range(len(all_context_ids)),
                        reverse_maps,
                        all_prompt_ids,
                    )
                ),
                reverse=True,
            )
        )
    )
    all_outputs = []

    stop_token_ids = [tokenizer.convert_tokens_to_ids(stop_token) for stop_token in args.stop_tokens]

    batch_idx = 0
    for batch in prange(math.ceil(len(all_context_ids) / args.batch_size)):
        batch_slice = (batch * args.batch_size, min((batch + 1) * args.batch_size, len(all_context_ids)))
        batch_size = batch_slice[1] - batch_slice[0]
        batch_context_tokens = all_context_ids[batch_slice[0] : batch_slice[1]]
        batch_reverse_maps = reverse_maps[batch_slice[0] : batch_slice[1]]
        batch_prompt_tokens = all_prompt_ids[batch_slice[0] : batch_slice[1]]

        if args.model_type == 'gpt2':
            batch_context_tensor = torch.tensor(
                model.pad_to_max_length(batch_context_tokens), dtype=torch.long, device=args.device
            )
            attention_mask = None
        else:
            padded_batch_context_tokens = []
            max_length = max([len(s) for s in batch_context_tokens])
            for i in range(len(batch_context_tokens)):
                padded_batch_context_tokens.append(
                    batch_context_tokens[i] + [pad_token_id] * (max_length - len(batch_context_tokens[i]))
                )
            batch_context_tensor = torch.tensor(padded_batch_context_tokens, dtype=torch.long, device=args.device)
            attention_mask = (batch_context_tensor != pad_token_id).to(torch.long)

        if args.model_type == 'mbart':
            decoder_start_token_id = tokenizer.lang_code_to_id[args.tgt_lang]
            model.config.decoder_start_token_id = decoder_start_token_id
        else:
            decoder_start_token_id = None

        if args.model_type == 'mbart50':
            forced_bos_token_id = tokenizer.lang_code_to_id[args.tgt_lang]
        else:
            forced_bos_token_id = None

        max_length = batch_context_tensor.shape[1] + args.length

        batch_outputs = [[] for _ in range(batch_size)]
        for hyperparameter_idx in range(len(args.temperature)):
            outputs = model.generate(
                input_ids=batch_context_tensor,
                bad_words_ids=None,
                attention_mask=attention_mask,
                decoder_start_token_id=decoder_start_token_id,
                forced_bos_token_id=forced_bos_token_id,
                min_length=args.min_output_length,
                max_length=max_length,
                num_beams=args.num_beams[hyperparameter_idx],
                top_k=args.top_k[hyperparameter_idx],
                top_p=args.top_p[hyperparameter_idx],
                early_stopping=True,
                num_return_sequences=args.num_samples[hyperparameter_idx],
                repetition_penalty=args.repetition_penalty[hyperparameter_idx],
                no_repeat_ngram_size=args.no_repeat_ngram_size[hyperparameter_idx],
                do_sample=args.temperature[hyperparameter_idx] != 0,
                temperature=args.temperature[hyperparameter_idx]
                if args.temperature[hyperparameter_idx] > 0
                else 1.0,  # if temperature==0, we do not sample
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
                output_attentions=output_attentions,
                return_dict_in_generate=True,
            )

            decoded = outputs.sequences
            cross_attentions = getattr(outputs, 'cross_attentions', None)

            if cross_attentions is not None:
                # stack tensors to shape (max_output_length, num_layers, batch_size, num_heads, 1, max_input_length)
                cross_attentions = torch.stack(([torch.stack(tuple) for tuple in cross_attentions]))

                # reshape to (num_layers, batch_size, num_heads, max_output_length, max_input_length)
                cross_attentions = cross_attentions.squeeze(4)
                cross_attentions = cross_attentions.permute(1, 2, 3, 0, 4).contiguous()

            if not isinstance(decoded, list):
                decoded = decoded[:, :].tolist()
            for i, out in enumerate(decoded):
                if 'bart' in args.model_type:
                    out = out[1:]  # remove </s> token at the beginning for bart, mbart, mbart50
                sample_index = (i // args.num_samples[hyperparameter_idx]) % batch_size
                if not args.output_prompt:
                    out = out[len(batch_prompt_tokens[sample_index]) :]
                min_index = len(out) - 1
                for stop_token_id in stop_token_ids + [eos_token_id]:
                    try:
                        index = out.index(stop_token_id)
                        min_index = min(index, min_index)
                    except ValueError:
                        pass

                ### include eos_token too; it will get removed during decoding
                min_index = min_index + 1
                out_cropped = out[:min_index]

                if args.task == 'translate' and cross_attentions is not None:
                    src_tokens = tokenizer.convert_ids_to_tokens(batch_context_tensor[sample_index])
                    tgt_tokens = tokenizer.convert_ids_to_tokens(out_cropped)

                    # get last layer attention vectors
                    # TODO: get penultimate layer of attention vectors
                    layer_attention = cross_attentions[-1, ...]
                    sample_layer_attention = layer_attention[sample_index, :, :, :]

                    if (
                        tgt_tokens[0] in [tokenizer.pad_token, special_tokens['bos_token'], special_tokens['sep_token']]
                        or (decoder_start_token_id and tgt_tokens[0] == tokenizer.id_to_lang_code[decoder_start_token_id])
                        or (forced_bos_token_id and tgt_tokens[0] == tokenizer.id_to_lang_code[forced_bos_token_id])
                    ):
                        # shift target tokens left to match the attention positions
                        tgt_tokens = tgt_tokens[1:]

                    while src_tokens[-1] == tokenizer.pad_token:
                        # remove all padding from src
                        src_tokens = src_tokens[:-1]
                    if src_tokens[-1] == special_tokens['sep_token']:
                        # remove trailing sep token
                        src_tokens = src_tokens[:-1]
                    if src_tokens[-1] == special_tokens['eos_token']:
                        # remove end token for better heatmap representation
                        src_tokens = src_tokens[:-1]

                    # remove language code from the beginning of src_tokens and shift layer_attention
                    len_prefix_wp = len(tokenizer.tokenize(model_input_prefix))
                    src_tokens = src_tokens[len_prefix_wp:]
                    sample_layer_attention = sample_layer_attention[:, :, len_prefix_wp:]

                    # crop to match src and tgt new lengths
                    sample_layer_attention = sample_layer_attention[:, : len(tgt_tokens), : len(src_tokens)]

                    sample_layer_attention_pooled = compute_attention(sample_layer_attention, args.att_pooling)

                    if args.plot_heatmaps:
                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        src_tokens = [token.lower() for token in src_tokens]
                        tgt_tokens = [token.lower() for token in tgt_tokens]
                        g = sns.heatmap(
                            torch.log(sample_layer_attention_pooled), xticklabels=src_tokens, yticklabels=tgt_tokens
                        )
                        g.set_xticklabels(g.get_xmajorticklabels(), fontsize=12)
                        g.set_yticklabels(g.get_ymajorticklabels(), fontsize=12)
                        if args.output_file is not None:
                            plt.savefig(
                                os.path.join(
                                    os.path.dirname(args.output_file), 'heatmap_{}'.format(batch_idx * batch_size + i)
                                )
                            )
                        plt.show()

                    # remove end token if present
                    if tgt_tokens[-1] in [special_tokens['bos_token'], special_tokens['eos_token']]:
                        tgt_tokens = tgt_tokens[:-1]

                    if args.replace_qp:
                        text, is_replaced = replace_quoted_params(
                            src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled
                        )
                        if not is_replaced and args.force_replace_qp:
                            text = force_replace_quoted_params(
                                src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled
                            )
                    else:
                        text = tokenizer.convert_tokens_to_string(tgt_tokens)
                else:
                    text = tokenizer.decode(out_cropped, clean_up_tokenization_spaces=False, skip_special_tokens=True)

                text = re.sub('\s\s+', ' ', text)  # remove duplicate white spaces
                text = text.strip()

                text = tokenize_cjk_chars(text)

                if not args.skip_heuristics:
                    text = output_heuristics(text, batch_reverse_maps[sample_index])
                batch_outputs[sample_index].append(text)

        all_outputs.extend(batch_outputs)
        if batch_idx == 0 and args.verbose:
            logger.info('First batch output: %s', str(all_outputs))
        batch_idx += 1

    # sort the results back to their original order
    _, all_outputs = tuple(zip(*sorted(list(zip(original_order, all_outputs)))))

    if args.output_file is not None:
        with open(args.output_file, 'w') as output_file:
            for i, output in enumerate(all_outputs):
                for j, text in enumerate(output):
                    # if num_samples is 1 keep the original id
                    if len(output) == 1:
                        id_ = all_example_ids[i]
                    else:
                        id_ = '{}-{}'.format(all_example_ids[i], j)
                    if args.output_example_ids_too:
                        output_file.write('\t'.join([id_, text]) + '\n')
                    else:
                        output_file.write(text + '\n')

    else:
        print(json.dumps(all_outputs, indent=2))

    metrics = compute_metrics(all_outputs, all_golds, reduction=args.metric_reduction)
    logger.info('Average BLEU score = %.2f', metrics['bleu'])
    logger.info('Exact match score = %.2f', metrics['em'])
