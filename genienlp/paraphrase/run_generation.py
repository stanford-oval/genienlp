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
import math
import json
import re
import copy
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# multiprocessing with CUDA
from torch.multiprocessing import Process, set_start_method

from genienlp.paraphrase.data_utils import create_features_from_tsv_file, output_heuristics
from genienlp.paraphrase.model_utils import compute_metrics

try:
     set_start_method('spawn')
except RuntimeError:
    pass
 
import torch
from transformers import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP
from .transformers_utils import BART_PRETRAINED_CONFIG_ARCHIVE_MAP, MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP, MARIAN_GROUP_MEMBERS, SPIECE_UNDERLINE

from transformers import GPT2Tokenizer

from .transformers_utils import BartForConditionalGeneration
from .transformers_utils import MarianMTModel

from transformers import BartTokenizer, MBartTokenizer
from transformers import MarianTokenizer
from transformers import PretrainedConfig
from ..util import set_seed, combine_files_on_disk, split_file_on_disk, get_part_path
from .GPT2Seq2Seq import GPT2Seq2Seq
from .data_utils import group_together


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

language_code_re = re.compile(">>.+<<")
ALL_MODELS = sum((tuple(map.keys()) for map in (GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, BART_PRETRAINED_CONFIG_ARCHIVE_MAP, MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2Seq2Seq, GPT2Tokenizer, {'sep_token': '<paraphrase>', 'end_token': '</paraphrase>'}),
    'bart': (BartForConditionalGeneration, BartTokenizer, {'sep_token': '<unk>', 'end_token': '</s>'}), # sep_token will not be used for BART
    'mbart': (BartForConditionalGeneration, MBartTokenizer, {'sep_token': '<unk>', 'end_token': '</s>'}), # sep_token will not be used for MBART
    'marian': (MarianMTModel, MarianTokenizer, {'sep_token': '<unk>', 'end_token': '</s>'}), # sep_token will not be used for MARIAN
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
    parser.add_argument('--id_column', type=int, default=None,
                        help='The column in the input file which contains the example ID.')
    parser.add_argument("--output_file", type=str, help="When specified, generated text will be written in this file. Defaults to stdout.")
    parser.add_argument("--intermediate_file", type=str, default='./paraphrase_tmp.tsv', help="Used to save intermediate results.")

    parser.add_argument('--output_prompt', action='store_true',
                        help='Whether we should include the prompt (specified via --prompt_column or --copy) in the output sequence')
    parser.add_argument("--length", type=int, default=50, help='The generated sentences will have a maximum length of len(input) + arg.length')
    parser.add_argument("--min_output_length", type=int, default=2, help='Will prevent stop tokens from appearing in the first --min_output_length tokens of the generated sentences.')
    parser.add_argument("--skip_heuristics", action='store_true', help='If True, will not replace special word such as NUMBER_0 in the input.')
    parser.add_argument("--is_cased", action='store_true',
                        help='If True, the trained model is cased, so if --skip_heuristics is not set, we will convert the input to upper case and the output back to lower case.')
    parser.add_argument("--metric_reduction", type=str, choices=['average', 'max'], default='average',
                        help="How we should calculate metrics where there are multiple generations per example.")
    
    parser.add_argument("--pipe_mode", action='store_true', help='If set, we will generate paraphrases of paraphrases of ... as well.')
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
    
    parser.add_argument('--trained_model_type', type=str, help='if provided we make sure the loaded model matches the model_type')
    
    parser.add_argument('--src_lang', type=str, default='en', help='source language used for translation task')
    parser.add_argument('--tgt_lang', type=str, help='target language used for translation task')
    parser.add_argument('--att_pooling', type=str, default='max', help='pooling used to calculate decoder-encoder attention values across different heads')
    parser.add_argument('--plot_heatmaps', action='store_true', help='whether to plot decoder-encoder attention heatmaps')
    parser.add_argument('--replace_qp', action='store_true', help='replace parameter values after translation with source values')
    parser.add_argument('--force_replace_qp', action='store_true', help='if we parameters could not be replaced leveraging quotation marks,'
                                                                        ' rely purely on attention to find text spans')
    parser.add_argument('--subsample', type=int, default=20000000, help='subsample input datasets')
    parser.add_argument('--task', type=str, required=True, choices=['paraphrase', 'translate'])
    parser.add_argument("--output_example_ids_too", action='store_true', help='Generate two column output with ids in the first column')


def main(args):
    hyperparameters = ['num_samples', 'temperature', 'top_k', 'top_p', 'repetition_penalty', 'num_beams', 'no_repeat_ngram_size']
    max_hyperparameter_len = max([len(getattr(args, h)) for h in hyperparameters])
    valid_len = [1, max_hyperparameter_len]
    for h in hyperparameters:
        if (len(getattr(args, h)) not in valid_len):
            logger.error('Hyperparameters should either have the same number of values as others or have exactly one value.')
        # If only one value is provided, use the same value for all samples
        setattr(args, h, getattr(args, h) * (max_hyperparameter_len // len(getattr(args, h))))

    logger.info('Will output %d sequences for each input.', sum(args.num_samples)if not args.pipe_mode else np.prod(args.num_samples))
    logger.info('Effective batch size for each GPU is %d', args.batch_size*max(args.num_samples))

    # TODO using intermediate files for pipe_mode is not clean. It needs to change.
    if args.pipe_mode:
        intermediate_files = [args.input_file] + [args.intermediate_file+str(i) for i in range(max_hyperparameter_len)]
        for i in range(max_hyperparameter_len):
            copy_args = copy.copy(args)
            for h in hyperparameters:
                setattr(copy_args, h, [getattr(args, h)[i]])
            copy_args.input_file = intermediate_files[i]
            copy_args.output_file = intermediate_files[i+1]
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
    config = PretrainedConfig.from_pretrained(args.model_name_or_path)
    
    # config.output_attentions = True
    # config.output_hidden_states = True
    
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
        raise ValueError('Model should be either GPT2, BART, MBART, or Marian')
    
    
    if args.trained_model_type and args.trained_model_type != '' and args.model_type != args.trained_model_type:
        raise ValueError('The loaded model type does not match with what the user provided')
    
    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 1)[1] in MARIAN_GROUP_MEMBERS:
        if not args.tgt_lang:
            raise ValueError('For translation task using Marian model, if target language is a group of languages, '
                             'you have to specify the --tgt_lang flag.')
        elif args.tgt_lang not in MARIAN_GROUP_MEMBERS[args.model_name_or_path.rsplit('-', 1)[1]]:
            raise ValueError('Target language is not in the model group languages, please specify the correct target language.')

    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 1)[1] not in MARIAN_GROUP_MEMBERS and args.tgt_lang:
        logger.warning('Target language should not be provided when using models with single language pairs,'
                       'otherwise the translation outputs will be incorrect; thus we ignore the target language you provided...')
        args.tgt_lang = None

    if args.prompt_column is not None and args.copy is not None and args.copy != 0:
        raise ValueError('Cannot copy from the input and use prompt at the same time. Disable either --copy or --prompt_column.')

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
    model = model_class.from_pretrained(args.model_name_or_path, output_attentions=True)
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

    all_input_sequences, all_input_sequence_lengths, all_example_ids, all_context_ids, estimated_output_lengths, all_golds, reverse_maps, all_prompt_ids = \
                                  create_features_from_tsv_file(file_path=args.input_file,
                                                                tokenizer=tokenizer,
                                                                input_column=args.input_column,
                                                                gold_column=args.gold_column,
                                                                id_column=args.id_column,
                                                                prompt_column=args.prompt_column,
                                                                copy=args.copy,
                                                                thingtalk_column=args.thingtalk_column,
                                                                sep_token_id=sep_token_id,
                                                                skip_heuristics=args.skip_heuristics,
                                                                is_cased=args.is_cased,
                                                                model_type=args.model_type,
                                                                src_lang=args.src_lang,
                                                                tgt_lang=args.tgt_lang,
                                                                subsample=args.subsample)

    # sort contexts based on their context length so that less generated tokens are thrown away and generation can be done faster
    estimated_output_lengths, all_input_sequence_lengths, all_input_sequences, all_context_ids, original_order, reverse_maps, all_prompt_ids = \
        tuple(zip(*sorted(list(zip(estimated_output_lengths, all_input_sequence_lengths, all_input_sequences, all_context_ids, range(len(all_context_ids)), reverse_maps, all_prompt_ids)), reverse=True)))
    all_outputs = []

    stop_token_ids = [tokenizer.convert_tokens_to_ids(stop_token) for stop_token in args.stop_tokens]
    
    batch_idx = 0
    for batch in tqdm(range(math.ceil(len(all_context_ids) / args.batch_size)), desc="Batch"):
        logging.info('') # to make kubectl properly print tqdm progress bar
        batch_slice = (batch*args.batch_size, min((batch+1)*args.batch_size, len(all_context_ids)))
        batch_size = batch_slice[1] - batch_slice[0]
        batch_input_sequences = all_input_sequences[batch_slice[0]: batch_slice[1]]
        batch_input_sequence_lengths = all_input_sequence_lengths[batch_slice[0]: batch_slice[1]]
        batch_context_tokens = all_context_ids[batch_slice[0]: batch_slice[1]]
        batch_reverse_maps = reverse_maps[batch_slice[0]: batch_slice[1]]
        batch_prompt_tokens = all_prompt_ids[batch_slice[0]: batch_slice[1]]

        if args.model_type == 'gpt2':
            batch_context_tensor = torch.tensor(model.pad_to_max_length(batch_context_tokens), dtype=torch.long, device=args.device)
            attention_mask = None
        else:
            padded_batch_context_tokens = []
            max_length = max([len(s) for s in batch_context_tokens])
            for i in range(len(batch_context_tokens)):
                padded_batch_context_tokens.append(batch_context_tokens[i]+[pad_token_id]*(max_length-len(batch_context_tokens[i])))
            batch_context_tensor = torch.tensor(padded_batch_context_tokens, dtype=torch.long, device=args.device)
            attention_mask = (batch_context_tensor!=pad_token_id).to(torch.long)

        all_encoder_attentions = None
        batch_outputs = [[] for _ in range(batch_size)]
        for hyperparameter_idx in range(len(args.temperature)):
            outputs = model.generate(input_ids=batch_context_tensor,
                                 bad_words_ids=None,
                                 attention_mask=attention_mask,
                                 min_length=args.min_output_length,
                                 # max_length=batch_context_tensor.shape[1]+args.length,
                                 max_length=args.length,
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
            
            if len(outputs) > 1:
                decoded, all_encoder_attentions = outputs
            else:
                decoded = outputs
                

            if not isinstance(decoded, list):
                decoded = decoded[:, :].tolist()
            for i, out in enumerate(decoded):
                sample_index = (i//args.num_samples[hyperparameter_idx]) % batch_size
                
                if not args.output_prompt:
                    out = out[len(batch_prompt_tokens[sample_index]):]
                min_index = len(out)-1
                for stop_token_id in stop_token_ids+[end_token_id]:
                    try:
                        index = out.index(stop_token_id)
                        min_index = min(index, min_index)
                    except ValueError:
                        pass

                min_index = min_index + 1
                out_cropped = out[:min_index]
            
                if args.task == 'translate':
                    src_tokens = tokenizer.convert_ids_to_tokens(batch_context_tensor[sample_index])
                    tgt_tokens = tokenizer.convert_ids_to_tokens(out_cropped)
                    
                    # get last layer attention vectors
                    layer_attention = all_encoder_attentions[-1]
                    sample_layer_attention = layer_attention[sample_index, :, :, :]

                    if tgt_tokens[0] == tokenizer.pad_token or tgt_tokens[0] == special_tokens['sep_token']:
                        # shift target tokens left to match the attention positions
                        tgt_tokens = tgt_tokens[1:]
                    while src_tokens[-1] == tokenizer.pad_token:
                        # remove all padding from src
                        src_tokens = src_tokens[:-1]
                    if src_tokens[-1] == special_tokens['sep_token']:
                        # remove trailing sep token
                        src_tokens = src_tokens[:-1]
                    if src_tokens[-1] == special_tokens['end_token']:
                        # remove end token for better heatmap representation
                        src_tokens = src_tokens[:-1]

                    if len(language_code_re.findall(src_tokens[0])):
                        # remove language code from the beginning of src_tokens and shift layer_attention
                        src_tokens = src_tokens[1:]
                        sample_layer_attention = sample_layer_attention[:, :, 1:]

                    # crop to match src and tgt new lengths
                    sample_layer_attention = sample_layer_attention[:, :len(tgt_tokens), :len(src_tokens)]

                    sample_layer_attention_pooled = compute_attention(sample_layer_attention, args.att_pooling)
                    
                    if args.plot_heatmaps:
                        sns.heatmap(torch.log(sample_layer_attention_pooled), xticklabels=src_tokens,
                                    yticklabels=tgt_tokens, annot=True)
                        if args.output_file is not None:
                            plt.savefig(os.path.join(os.path.dirname(args.output_file),
                                                     'heatmap_{}'.format(batch_idx * batch_size + i)))
                        plt.show()
                    
                    if args.replace_qp:
                        text, is_replaced = replace_quoted_params(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled, args.model_type)
                        if not is_replaced and args.force_replace_qp:
                            text = force_replace_quoted_params(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled, args.model_type)
                    else:
                        text = tokenizer.convert_tokens_to_string(tgt_tokens)
                else:
                    text = tokenizer.decode(out_cropped, clean_up_tokenization_spaces=True, skip_special_tokens=True)

                text = re.sub('\s\s+', ' ', text)  # remove duplicate white spaces
                text = text.strip()
                if not args.skip_heuristics:
                    text = output_heuristics(text, batch_reverse_maps[sample_index])
                batch_outputs[sample_index].append(text)
                
        all_outputs.extend(batch_outputs)
        if batch_idx < 1:
            logger.info('First batch output: %s', str(all_outputs))
        batch_idx += 1

    # sort the results back to their original order
    _, all_outputs = tuple(zip(*sorted(list(zip(original_order, all_outputs)))))

    if args.output_file is not None:
        with open(args.output_file, 'w') as output_file:
            for i, output in enumerate(all_outputs):
                for j, text in enumerate(output):
                    if args.output_example_ids_too:
                        output_file.write('\t'.join(['{}-{}'.format(all_example_ids[i], j), text]) + '\n')
                    else:
                        output_file.write(text + '\n')
    else:
        print(json.dumps(all_outputs, indent=2))

    metrics = compute_metrics(all_outputs, all_golds, reduction=args.metric_reduction)
    logger.info('Average BLEU score = %.2f', metrics['bleu'])
    logger.info('Exact match score = %.2f', metrics['em'])


def compute_attention(sample_layer_attention, att_pooling):
    sample_layer_attention_pooled = None
    if att_pooling == 'mean':
        sample_layer_attention_pooled = torch.mean(sample_layer_attention, dim=0, keepdim=False)
    elif att_pooling == 'max':
        sample_layer_attention_pooled = torch.max(sample_layer_attention, dim=0, keepdim=False)[0]
        
    return sample_layer_attention_pooled

def replace_quoted_params(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled, model_type):
    # find positions of quotation marks in src and tgt
    src2tgt_mapping = {}
    src2tgt_mapping_index = {}
    
    ## FIXED: quotation marks are exclusively used to wrap parameters so just check if they are present in target token
    # quote_wordpiece = tokenizer.tokenize('"')[0]
    # quote_token = '"'
    
    src_spans_ind = [index for index, token in enumerate(src_tokens) if '"' in token]
    tgt_spans_ind = [index for index, token in enumerate(tgt_tokens) if '"' in token]
    
    if model_type == 'marian':
        src_strings = tokenizer.spm_source.DecodePieces(src_tokens)
        tgt_strings = tokenizer.spm_target.DecodePieces(tgt_tokens)
    else:
        src_strings = tokenizer.convert_tokens_to_string(src_tokens)
        tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)

    
    if len(src_spans_ind) % 2 != 0:
        logging.error('corrupted span in src string: [{}]'.format(src_strings))
    if len(tgt_spans_ind) % 2 != 0:
        logging.error('corrupted span in tgt string: [{}] with src string: [{}]\n'
                      'outputting example without reverting the parameter'.format(tgt_strings, src_strings))
    
        return tgt_strings, False
    
    # arrange spans and exclude quotation mark indices
    src_spans = [(src_spans_ind[i] + 1, src_spans_ind[i + 1] - 1) for i in range(0, len(src_spans_ind), 2)]
    tgt_spans = [(tgt_spans_ind[i] + 1, tgt_spans_ind[i + 1] - 1) for i in range(0, len(tgt_spans_ind), 2)]
    
    if len(src_spans) != len(tgt_spans):
        logging.error('numbers of spans in src and tgt strings do not match: [{}], [{}]\n'
                      'outputting example without reverting the parameter'.format(src_strings, tgt_strings))
            
        return tgt_strings, False
    
    tgt_span_success = set()
    for src_idx, (beg, end) in enumerate(src_spans):
        i = beg
        tgt_span_idx = None
        while i <= end:
            max_tgt_att_idx = torch.argmax(sample_layer_attention_pooled[:, i]).item()
            
            # find span in tgt that contains this index
            for tgt_idx, (s1, s2) in enumerate(tgt_spans):
                if s1 <= max_tgt_att_idx <= s2 and (s1, s2) not in tgt_span_success:
                    tgt_span_idx = tgt_idx
                    src2tgt_mapping[(beg, end)] = (s1, s2)
                    src2tgt_mapping_index[src_idx] = tgt_span_idx
                    tgt_span_success.add((s1, s2))
                    break
            if tgt_span_idx is not None:
                break
            else:
                # span could not be found; check the next wordpiece
                i += 1
        
        if tgt_span_idx is None:
            logger.error('Could not find a corresponding span in tgt for ({}, {}) src span in src string: [{}]'.format(beg, end, src_strings))
            return tgt_strings, False
    ####
    # replacing in word-piece space is not clean since Marian uses different spm models for src and tgt
    ####
    # # replace property values (wrapped in quotation marks) in target text with source values
    # tgt2src_mapping = {v: k for k, v in src2tgt_mapping.items()}
    # tgt_begin2span = {k[0]: k for k, v in tgt2src_mapping.items()}
    # all_tgt_begins = set(tgt_begin2span.keys())
    #
    # new_tgt_tokens = []
    # i = 0
    # while i < len(tgt_tokens):
    #     if i in all_tgt_begins:
    #         tgt_span = tgt_begin2span[i]
    #         src_span = tgt2src_mapping[tgt_span]
    #         new_tgt_tokens.extend(src_tokens[src_span[0]: src_span[1]+1])
    #         i += tgt_span[1] - tgt_span[0] + 1
    #     else:
    #         new_tgt_tokens.append(tgt_tokens[i])
    #         i += 1
    # final_output = tokenizer.convert_tokens_to_ids(new_tgt_tokens)
    
    quoted_pattern_maybe_space = re.compile(r'\"\s?([^"]*?)\s?\"')
    
    src_matches = list(re.finditer(quoted_pattern_maybe_space, src_strings))
    tgt_matches = list(re.finditer(quoted_pattern_maybe_space, tgt_strings))
    
    tgt2src_mapping_index = {v: k for k, v in src2tgt_mapping_index.items()}

    # move through characters
    tokens = []
    curr = 0
    for pos, match in enumerate(tgt_matches):
        start, end = match.span()
        if start > curr:
            tokens.append(tgt_strings[curr:start])
        replace_match = src_matches[tgt2src_mapping_index[pos]]
        tokens.append(replace_match.group(0))
        curr = end
    if curr < len(tgt_strings):
        tokens.append(tgt_strings[curr:])
    
    text = ' '.join(tokens)
    
    return text, True


def force_replace_quoted_params(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled, model_type):
    # find positions of quotation marks in src and tgt
    src2tgt_mapping = {}
    
    src_spans_ind = [index for index, token in enumerate(src_tokens) if '"' in token]
    
    if model_type == 'marian':
        src_strings = tokenizer.spm_source.DecodePieces(src_tokens)
        tgt_strings = tokenizer.spm_target.DecodePieces(tgt_tokens)
    else:
        src_strings = tokenizer.convert_tokens_to_string(src_tokens)
        tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)
        

    tgt_is_piece = [1 if token[0] == SPIECE_UNDERLINE else 0 for token in tgt_tokens]
    tgt_piece2word_mapping = list(np.cumsum(tgt_is_piece) - 1)
    
    if len(src_spans_ind) % 2 != 0:
        logging.error('corrupted span in src string: [{}]'.format(src_strings))
    
    # arrange spans and exclude quotation mark indices
    src_spans = [(src_spans_ind[i] + 1, src_spans_ind[i + 1] - 1) for i in range(0, len(src_spans_ind), 2)]
    
    for src_idx, (beg, end) in enumerate(src_spans):
        # check wordpiece after beg and before end
        s1 = torch.argmax(sample_layer_attention_pooled[:, beg]).item()
        s2 = torch.argmax(sample_layer_attention_pooled[:, end]).item()
        
        src2tgt_mapping[(beg, end)] = (s1, s2)
    

    quoted_pattern_maybe_space = re.compile(r'\"\s?([^"]*?)\s?\"')
    
    src_matches = list(re.finditer(quoted_pattern_maybe_space, src_strings))
    
    # update src2tgt_mapping to map to word indices in response
    for key, value in src2tgt_mapping.items():
        s1, s2 = value
        src2tgt_mapping[key] = tgt_piece2word_mapping[s1] - 1, tgt_piece2word_mapping[s2] + 1
    
    # move through words
    tgt_strings_words = tgt_strings.split(' ')
    tokens = []
    curr = 0
    for i , (key, value) in enumerate(src2tgt_mapping.items()):
        start, end = value
        if start > curr:
            tokens.extend(tgt_strings_words[curr:start])
        replace_match = src_matches[i]
        tokens.append(replace_match.group(0))
        curr = end
    if curr < len(tgt_strings_words):
        tokens.extend(tgt_strings_words[curr:])
    
    text = ' '.join(tokens)
    
    return text
