# Parts of this file were adopted from https://github.com/huggingface/transformers.
# See the original copyright notice below.

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
This script is used for fine-tuning library models on a dataset.
GPT and GPT-2 are fine-tuned using a Causal Language Modeling (CLM) loss while BERT family models are fine-tuned
using a Masked Language Modeling (MLM) loss. BART, MBART, and MARIAN are fine-tuned on supervised data using Cross Entropy (CE) loss.
"""

from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import shutil

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
    MBartConfig,
    MBartForConditionalGeneration,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from ..data_utils.progbar import prange, progress_bar
from ..model_utils.transformers_utils import GenieMBartTokenizer
from ..util import set_seed, split_file_on_disk
from .data_utils import add_special_tokens, mask_tokens
from .dataset import LengthSortedSampler, TextDataset
from .model_utils import (
    _rotate_checkpoints,
    check_args,
    freeze_embeds,
    freeze_params,
    get_transformer_schedule_with_warmup,
    shift_tokens_right,
)

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'camembert': (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
    'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
    'mbart': (MBartConfig, MBartForConditionalGeneration, GenieMBartTokenizer),
    'marian': (MarianConfig, MarianMTModel, MarianTokenizer),
}


def train(
    args,
    train_dataset,
    model,
    tokenizer,
    input_file_name=None,
    multiple_shards=False,
    init_global_step=0,
    init_epochs_trained=0,
):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.sort_by_length:
        train_sampler = LengthSortedSampler(
            train_dataset, batch_size=args.train_batch_size * args.gradient_accumulation_steps, shuffle=True
        )
    else:
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=train_dataset.collate_fn
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    elif args.scheduler == 'transformer':
        if args.model_type == 'bert':
            dimension = model.config.hidden_size
        elif args.model_type == 'gpt2':
            dimension = model.config.n_embd
        else:
            logger.error('Cannot detect hidden size dimensions in this model type. Config: %s', model.config)
        scheduler = get_transformer_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total, dimension=dimension
        )
    else:
        logger.error('Unknown scheduler type.')

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')) and os.path.isfile(
        os.path.join(args.model_name_or_path, 'scheduler.pt')
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Input file name = %s", input_file_name)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = init_global_step
    epochs_trained = init_epochs_trained
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split('-')[-1].split('/')[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = (
            global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        ) * args.gradient_accumulation_steps

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0, 0

    model_to_resize = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    if multiple_shards:
        train_iterator = prange(epochs_trained, 1, desc="Epoch", disable=args.local_rank not in [-1, 0])
    else:
        train_iterator = prange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
        )
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    best_eval_perplexity = float('Inf')
    for _ in train_iterator:
        if args.max_steps > 0 and not multiple_shards:
            total_steps = args.max_steps * args.gradient_accumulation_steps
        else:
            total_steps = len(train_dataloader)
        epoch_iterator = progress_bar(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], total=total_steps
        )
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, attention_mask, labels, position_ids, segment_ids = batch

            if args.mlm:
                inputs, labels = mask_tokens(inputs, labels, tokenizer, args.mlm_probability, args.mlm_ignore_index)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            attention_mask = attention_mask.to(args.device)
            position_ids = position_ids.to(args.device)
            segment_ids = segment_ids.to(args.device)
            model.train()

            model_inputs = {'input_ids': inputs, 'use_cache': False}

            # prepare inputs for mbart, and marian
            if args.model_type in ['mbart', 'marian']:
                model_inputs['attention_mask'] = attention_mask
                decoder_input_ids = shift_tokens_right(labels, args.mlm_ignore_index)
                decoder_input_ids[decoder_input_ids == args.mlm_ignore_index] = tokenizer.pad_token_id
                model_inputs['decoder_input_ids'] = decoder_input_ids
            elif args.model_type == 'bart':
                # TODO according to huggingface bart should also use shift_tokens_right
                # check if that change affects results
                model_inputs['attention_mask'] = attention_mask
                decoder_input_ids = labels.contiguous()
                decoder_input_ids[decoder_input_ids == args.mlm_ignore_index] = tokenizer.pad_token_id
                model_inputs['decoder_input_ids'] = decoder_input_ids
            else:
                model_inputs.update({'position_ids': position_ids, 'token_type_ids': segment_ids})

            outputs = model(**model_inputs)
            lm_logits = outputs.logits.contiguous()
            assert lm_logits.shape[-1] == model.config.vocab_size

            # CrossEntropyLoss ignore_index defaults to -100
            # If a different mlm_ignore_index is provided we make sure it is ignored when calculating the loss
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=args.mlm_ignore_index)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if args.local_rank in [-1, 0] and (
                (args.logging_steps > 0 and global_step % args.logging_steps == 0 and global_step != 0)
                or step == total_steps - 1
            ):
                # Log metrics
                if (
                    args.local_rank == -1 and args.evaluate_during_training
                ):  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, tokenizer)
                    if args.aux_eval_data_file is not None:
                        aux_results = evaluate(args, model, tokenizer, aux=True)
                        for key, value in aux_results.items():
                            tb_writer.add_scalar('auxiliary_eval_{}'.format(key), value, global_step)
                    if best_eval_perplexity > results['perplexity']:
                        best_eval_perplexity = results['perplexity']
                        if not os.path.exists(args.output_dir):
                            os.makedirs(args.output_dir)
                        logger.info("Saving new best model to %s", args.output_dir)
                        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                        # They can then be reloaded using `from_pretrained()`
                        model_to_save = (
                            model.module if hasattr(model, 'module') else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)

                        # Good practice: save your training arguments together with the trained model
                        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                # TODO add generated text to tensorboard
                # tb_writer.add_text('eval/generated_text', gen_text, global_step)
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            if (
                args.local_rank in [-1, 0]
                and args.save_steps > 0
                and global_step % args.save_steps == 0
                and global_step != 0
                and args.save_total_limit > 0
            ):
                checkpoint_prefix = 'checkpoint'
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, 'module') else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                _rotate_checkpoints(args, checkpoint_prefix)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss


def evaluate(args, model, tokenizer, prefix="", aux=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if aux:
        file_path = args.aux_eval_data_file
    else:
        file_path = args.eval_data_file

    eval_dataset = TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    if args.sort_by_length:
        eval_sampler = LengthSortedSampler(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
    else:
        eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in progress_bar(eval_dataloader, desc="Evaluating"):
        inputs, attention_mask, labels, position_ids, segment_ids = batch
        if args.mlm:
            inputs, labels = mask_tokens(inputs, labels, tokenizer, args.mlm_probability, args.mlm_ignore_index)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)
        position_ids = position_ids.to(args.device)
        segment_ids = segment_ids.to(args.device)

        with torch.no_grad():
            model_inputs = {'input_ids': inputs, 'use_cache': False}

            if args.model_type in ['mbart', 'marian']:
                model_inputs['attention_mask'] = attention_mask
                decoder_input_ids = shift_tokens_right(labels, args.mlm_ignore_index)
                decoder_input_ids[decoder_input_ids == args.mlm_ignore_index] = tokenizer.pad_token_id
                model_inputs['decoder_input_ids'] = decoder_input_ids
            elif args.model_type == 'bart':
                model_inputs['attention_mask'] = attention_mask
                decoder_input_ids = labels.contiguous()
                decoder_input_ids[decoder_input_ids == args.mlm_ignore_index] = tokenizer.pad_token_id
                model_inputs['decoder_input_ids'] = decoder_input_ids
            else:
                model_inputs.update({'position_ids': position_ids, 'token_type_ids': segment_ids})

            outputs = model(**model_inputs)
            lm_logits = outputs.logits

            # debugging
            if args.debug:
                print()
                print([tokenizer.decode(t, skip_special_tokens=True) for t in lm_logits.max(-1)[1]])

            assert lm_logits.shape[-1] == model.config.vocab_size

            tokenizer.batch_decode(lm_logits.max(-1)[1])

            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=args.mlm_ignore_index)
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            eval_loss += loss.mean().item()

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def parse_argv(parser):
    ## Required parameters
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--tensorboard_dir", default=None, type=str, help="The output directory where the tensorboard files will be written."
    )
    parser.add_argument("--train_data_file", default=None, type=str, help="The input training data file.")
    parser.add_argument(
        "--aux_train_data_file", default=None, type=str, help="An input training data file for the target domain."
    )
    parser.add_argument(
        '--start_special_token', type=str, default='<paraphrase>', help='The special token for the start of paraphrases.'
    )
    parser.add_argument(
        '--end_special_token', type=str, default='</paraphrase>', help='The special token for the end of paraphrases.'
    )
    parser.add_argument('--pad_token', type=str, default='<pad>', help='The special token for padding..')
    parser.add_argument(
        '--train_all_tokens',
        action='store_true',
        help='If True, the model will be trained on input and output sequences, as opposed to only tokens of the output sequence',
    )
    parser.add_argument(
        "--reverse_position_ids",
        action='store_true',
        help='If we assume we know the length of the output sequence beforehand, we can do a better job at generation.',
    )
    parser.add_argument('--subsample', default=20000000, type=int, help='subsample the datasets')
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--aux_eval_data_file",
        default=None,
        type=str,
        help="An additional input evaluation data file to evaluate the perplexity on (a text file).",
    )

    parser.add_argument("--model_type", default="bert", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument(
        "--model_name_or_path", default="bert-base-cased", type=str, help="The model checkpoint for weights initialization."
    )

    parser.add_argument(
        "--mlm", action='store_true', help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--mlm_ignore_index",
        type=int,
        default=-100,
        help="Tokens with this label will be ignore when calculating masked language loss",
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default=".embeddings",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training examples that are longer than this size (input length + output_length) will not be used for training or evaluation."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        '--sort_by_length',
        action='store_true',
        help='Sorts the training set by example length (input length + output_length) to reduce padding and speed up training. Has no effect on accuracy.',
    )
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action='store_true', help="Run evaluation during training at each logging step."
    )
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--scheduler",
        default='linear',
        type=str,
        choices=['linear', 'transformer'],
        help="The type of learning rate scheduler to use.",
    )

    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        '--save_total_limit',
        type=int,
        default=None,
        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default',
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action='store_true',
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        '--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument('--src_lang', type=str, help='source language used for translation task')
    parser.add_argument('--tgt_lang', type=str, help='target language used for translation task')

    parser.add_argument('--debug', action='store_true', help='print intermediate results for debugging')
    parser.add_argument('--no_pretraining', action='store_true', help='Remove all pre-training and train a model from scratch')
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_embeds", action="store_true")

    parser.add_argument(
        '--input_column', type=int, default=0, help='The column in the input file which contains the input sentences.'
    )
    parser.add_argument(
        '--gold_column',
        type=int,
        default=1,
        help='The column in the input file which contains the gold sentences. Defaults to --input_column if no gold is available.',
    )

    parser.add_argument(
        '--cache_input_data', action='store_true', help='Cache examples from input data for faster subsequent trainings'
    )

    parser.add_argument(
        '--num_input_chunks',
        default=1,
        type=int,
        help='We split input into multiple chunks, then load and train on each chunk individually',
    )
    parser.add_argument('--delete_after_chunking', action='store_true', help='Delete input file after chunking it')

    parser.add_argument('--no_fast_tokenizer', action='store_true', help='Use slow version of huggingface tokenizer')


def main(args):
    if args.model_type == 'bert' and (
        args.pad_token != '[PAD]' or args.start_special_token != '[SEP]' or args.end_special_token != '[SEP]'
    ):
        raise ValueError("BERT already has its own special tokens [PAD] and [SEP]. You should use them for better results.")
    if args.do_train:
        if args.train_data_file is None:
            raise ValueError(
                "Cannot do training without a training data file. Either supply a file to --train_data_file "
                "or remove the --do_train argument."
            )
        if args.tensorboard_dir is None:
            raise ValueError("Cannot do training without specifying --tensorboard_dir")

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        # clean all files within the directory
        if args.overwrite_output_dir:
            shutil.rmtree(args.output_dir)
        else:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir
                )
            )

    check_args(args)

    if args.gold_column is None:
        args.gold_column = args.input_column

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, cache_dir=args.cache_dir if args.cache_dir else None
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=not args.no_fast_tokenizer,
    )
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if args.no_pretraining:
        # only load model architecture but not the weights
        model = model_class(config)
    else:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    add_special_tokens(
        model,
        tokenizer,
        additional_special_tokens=[args.start_special_token, args.end_special_token],
        pad_token=args.pad_token,
    )
    model.to(args.device)

    if args.freeze_embeds:
        freeze_embeds(model)
    if args.freeze_encoder:
        freeze_params(model.get_encoder())
    if args.freeze_decoder:
        if args.model_type in ['bart', 'mbart', 'marian']:
            freeze_params(model.model.decoder)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

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
    args.model_input_prefix = model_input_prefix

    # Training
    if args.do_train:

        if args.num_input_chunks > 1:

            all_input_files = split_file_on_disk(
                args.train_data_file, args.num_input_chunks, delete=args.delete_after_chunking
            )
            global_step, epochs_trained, total_tr_loss = 0, 0, 0.0

            for n in range(args.num_train_epochs):

                for i in range(args.num_input_chunks):

                    if args.local_rank not in [-1, 0]:
                        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

                    train_dataset = TextDataset(
                        tokenizer, args, file_path=all_input_files[i], block_size=args.block_size, evaluate=True
                    )

                    if args.local_rank == 0:
                        torch.distributed.barrier()

                    global_step, tr_loss = train(
                        args, train_dataset, model, tokenizer, all_input_files[i], True, global_step, epochs_trained
                    )
                    total_tr_loss += tr_loss

                epochs_trained += 1

                logger.info(" global_step = %s, average loss = %s", global_step, total_tr_loss / global_step)

            for file in all_input_files:
                os.remove(file)

        else:
            if args.local_rank not in [-1, 0]:
                torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

            train_dataset = TextDataset(
                tokenizer, args, file_path=args.train_data_file, block_size=args.block_size, evaluate=True
            )

            if args.local_rank == 0:
                torch.distributed.barrier()

            global_step, tr_loss = train(args, train_dataset, model, tokenizer, args.train_data_file, False)

            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results
