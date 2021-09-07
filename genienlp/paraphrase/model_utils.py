import glob
import logging
import math
import os
import re
import shutil

import numpy as np
import torch

from ..metrics import computeBLEU
from ..model_utils.transformers_utils import MARIAN_GROUP_MEMBERS
from ..util import FAIRSEQ_LANGUAGE_CODES, get_mbart_lang

logger = logging.getLogger(__name__)


def shift_tokens_right(input_ids, pad_token_id):
    """
    Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
    Adopted from huggingface's finetune.py code
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def freeze_params(model):
    for par in model.parameters():
        par.requires_grad = False


def unfreeze_params(model):
    for par in model.parameters():
        par.requires_grad = True


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    if hasattr(model, 'model'):
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)


def unfreeze_embeds(model):
    """Unfreeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    if hasattr(model, 'model'):
        unfreeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            unfreeze_params(d.embed_positions)
            unfreeze_params(d.embed_tokens)
    else:
        unfreeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            unfreeze_params(d.embed_tokens)


def check_args(args):
    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 1)[1] in MARIAN_GROUP_MEMBERS:
        if not args.tgt_lang:
            raise ValueError(
                'For translation task using Marian model, if target language is a group of languages, '
                'you have to specify the --tgt_lang flag.'
            )
        elif args.tgt_lang not in MARIAN_GROUP_MEMBERS[args.model_name_or_path.rsplit('-', 1)[1]]:
            if args.tgt_lang == 'pl':
                args.tgt_lang = 'pol'
            elif args.tgt_lang == 'fa':
                args.tgt_lang = 'pes'
            else:
                raise ValueError(
                    'Target language is not in the model group languages, please specify the correct target language.'
                )

    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 2)[1] in MARIAN_GROUP_MEMBERS:
        if not args.src_lang:
            raise ValueError(
                'For translation task using Marian model, if source language is a group of languages, '
                'you have to specify the --src_lang flag.'
            )
        elif args.src_lang not in MARIAN_GROUP_MEMBERS[args.model_name_or_path.rsplit('-', 2)[1]]:
            if args.src_lang == 'pl':
                args.src_lang = 'pol'
            elif args.src_lang == 'fa':
                args.src_lang = 'pes'
            raise ValueError(
                'Source language is not in the model group languages, please specify the correct source language.'
            )

    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 1)[1] not in MARIAN_GROUP_MEMBERS and args.tgt_lang:
        logger.warning(
            'Target language should not be provided when using Marian models with single target language,'
            ' otherwise the translation outputs will be incorrect; thus we ignore the target language you provided...'
        )
        args.tgt_lang = None

    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 2)[1] not in MARIAN_GROUP_MEMBERS and args.src_lang:
        logger.warning(
            'Source language should not be provided when using Marian models with single source language,'
            ' otherwise the translation outputs will be incorrect; thus we ignore the source language you provided...'
        )
        args.src_lang = None

    if args.model_type == 'mbart' and not (args.tgt_lang and args.src_lang):
        raise ValueError('Source and Target language should be provided when using mBART cc25 model')

    # adjust language ids for mbart models
    if args.model_type in ['mbart', 'mbart50']:
        if args.src_lang not in FAIRSEQ_LANGUAGE_CODES:
            args.src_lang = get_mbart_lang(args.src_lang)
        if args.tgt_lang not in FAIRSEQ_LANGUAGE_CODES:
            args.tgt_lang = get_mbart_lang(args.tgt_lang)


def sort_checkpoints(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, "checkpointepoch=*.ckpt"), recursive=True)))


def get_transformer_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, dimension):
    num_warmup_steps = max(1, num_warmup_steps)

    def lr_lambda(current_step):
        current_step += 1
        return (
            1.0
            / math.sqrt(dimension)
            * min(1 / math.sqrt(current_step), current_step / (num_warmup_steps * math.sqrt(num_warmup_steps)))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


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
        if reduction == 'average':
            bleu_score /= len(output)
            exact_match /= len(output)
        total_bleu += bleu_score
        total_exact_match += exact_match
        count += 1

    return {'bleu': total_bleu / count, 'em': total_exact_match / count * 100}


LOG_EVERY = 5000
log_counter = 0


def do_log(counter):
    return not counter % LOG_EVERY


def replace_quoted_params(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled):
    # find positions of quotation marks in src and tgt
    src2tgt_mapping = {}
    src2tgt_mapping_index = {}
    global log_counter

    # Note: quotation marks are exclusively used to wrap parameters so just check if they are present in the target sentence
    src_quotation_symbols = ['"']
    tgt_quotation_symbols = ['"', '«', '»', '“', '„']

    tokenizer._decode_use_source_tokenizer = False

    tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)
    for symbol in tgt_quotation_symbols:
        # 1) replace multiple quotes with single quote
        tgt_strings = re.sub(rf"{symbol}+", rf"{symbol}", tgt_strings)

        # 2) add space around every symbol
        tgt_strings = re.sub(rf"{symbol}", rf" {symbol} ", tgt_strings)

        # 3) remove any double spaces
        tgt_strings = re.sub(r"\s\s", " ", tgt_strings)

    with tokenizer.as_target_tokenizer():
        tgt_tokens = tokenizer.tokenize(tgt_strings)

    src_spans_ind = [
        index for index, token in enumerate(src_tokens) if any([symbol in token for symbol in src_quotation_symbols])
    ]
    tgt_spans_ind = [
        index for index, token in enumerate(tgt_tokens) if any([symbol in token for symbol in tgt_quotation_symbols])
    ]

    tokenizer._decode_use_source_tokenizer = True
    src_strings = tokenizer.convert_tokens_to_string(src_tokens)
    tokenizer._decode_use_source_tokenizer = False
    tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)

    if len(src_spans_ind) % 2 != 0:
        if do_log(log_counter):
            logging.error(f'Corrupted span in src string: [{src_strings}]')
        log_counter += 1
        return tgt_strings, False
    if len(tgt_spans_ind) % 2 != 0:
        if do_log(log_counter):
            logging.error(f'Corrupted span in tgt string: [{tgt_strings}] with src string: [{src_strings}]\n')
        log_counter += 1
        return tgt_strings, False

    # arrange spans and exclude quotation mark indices
    src_spans = [(src_spans_ind[i] + 1, src_spans_ind[i + 1] - 1) for i in range(0, len(src_spans_ind), 2)]
    tgt_spans = [(tgt_spans_ind[i] + 1, tgt_spans_ind[i + 1] - 1) for i in range(0, len(tgt_spans_ind), 2)]

    if len(src_spans) != len(tgt_spans):
        if do_log(log_counter):
            logging.error(f'Numbers of spans in tgt and src strings do not match: [{tgt_strings}], [{src_strings}]\n')
        log_counter += 1
        return tgt_strings, False

    tgt_span_success = set()
    for src_idx, (beg, end) in enumerate(src_spans):
        i = beg
        tgt_span_idx = None
        while i <= end:
            max_tgt_att_idx = torch.argmax(sample_layer_attention_pooled[:, i]).item()

            # find span in tgt that contains this index; -1 and +1 to include target quotations marks
            for tgt_idx, (s1, s2) in enumerate(tgt_spans):
                if s1 - 1 <= max_tgt_att_idx <= s2 + 1 and (s1, s2) not in tgt_span_success:
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
            if do_log(log_counter):
                logger.error(
                    f'Could not find a corresponding span in tgt for ({beg}, {end}) src span in src string: [{src_strings}]'
                )
            log_counter += 1
            return tgt_strings, False

    src_quoted_pattern_maybe_space = re.compile(r'[{0}]\s?([^{0}]*?)\s?[{0}]'.format(''.join(src_quotation_symbols)))
    tgt_quoted_pattern_maybe_space = re.compile(r'[{0}]\s?([^{0}]*?)\s?[{0}]'.format(''.join(tgt_quotation_symbols)))

    src_matches = list(re.finditer(src_quoted_pattern_maybe_space, src_strings))
    tgt_matches = list(re.finditer(tgt_quoted_pattern_maybe_space, tgt_strings))

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


def force_replace_quoted_params(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled):
    # find positions of quotation marks in src
    src2tgt_mapping = {}

    src_quotation_symbols = ['"']
    tgt_quotation_symbols = ['"', '«', '»', '“', '„']

    global log_counter

    # replace double quotes with single quote
    for symbol in tgt_quotation_symbols:
        tgt_tokens = ' '.join(tgt_tokens).replace(f'{symbol}{symbol}', f'{symbol}').split(' ')

    src_spans_ind = [
        index for index, token in enumerate(src_tokens) if any([symbol in token for symbol in src_quotation_symbols])
    ]
    tgt_is_not_piece = [int(not tokenizer.is_piece_fn(token)) for token in tgt_tokens]
    tgt_piece2word_mapping = list(np.cumsum(tgt_is_not_piece) - 1)

    if len(src_spans_ind) % 2 != 0:
        if do_log(log_counter):
            tokenizer._decode_use_source_tokenizer = True
            logging.error(f'Corrupted span in src string: [{tokenizer.convert_tokens_to_string(src_tokens)}]')
            tokenizer._decode_use_source_tokenizer = False
        log_counter += 1

    tokenizer._decode_use_source_tokenizer = True
    src_strings = tokenizer.convert_tokens_to_string(src_tokens)
    tokenizer._decode_use_source_tokenizer = False
    tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)

    # arrange spans but DO NOT exclude quotation mark indices
    src_spans = [(src_spans_ind[i], src_spans_ind[i + 1]) for i in range(0, len(src_spans_ind), 2)]

    for src_idx, (beg, end) in enumerate(src_spans):
        s1 = torch.argmax(sample_layer_attention_pooled[:, beg]).item()
        s2 = torch.argmax(sample_layer_attention_pooled[:, end]).item()

        # clamp values to max tgt_tokens length
        s1 = min(s1, len(tgt_tokens) - 1)
        s2 = min(s2, len(tgt_tokens) - 1)

        # switch tgt begin and end indices
        if s1 > s2:
            s1, s2 = s2, s1

        src2tgt_mapping[(beg, end)] = (s1, s2)

    quoted_pattern_maybe_space = re.compile(r'\"\s?([^"]*?)\s?\"')

    src_matches = list(re.finditer(quoted_pattern_maybe_space, src_strings))

    # update src2tgt_mapping to map to word indices in response
    for key, value in src2tgt_mapping.items():
        s1, s2 = value
        # try:
        src2tgt_mapping[key] = (
            max(0, tgt_piece2word_mapping[s1]),
            min(tgt_piece2word_mapping[s2], len(tgt_tokens)),
        )

    # move through words
    tgt_strings_words = tgt_strings.split(' ')
    tokens = []
    curr = 0
    for i, (key, value) in enumerate(src2tgt_mapping.items()):
        start, end = value
        if start > curr:
            tokens.extend(tgt_strings_words[curr:start])
        replace_match = src_matches[i]
        tokens.append(replace_match.group(0))
        # +1 since it's inclusive
        curr = end + 1
    if curr < len(tgt_strings_words):
        tokens.extend(tgt_strings_words[curr:])

    text = ' '.join(tokens)

    return text
