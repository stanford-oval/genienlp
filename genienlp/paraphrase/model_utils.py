import torch
import math
import os
import glob
import re
import logging
import shutil
import numpy as np

from .transformers_utils import SPIECE_UNDERLINE, MARIAN_GROUP_MEMBERS
from transformers.models.mbart.tokenization_mbart import FAIRSEQ_LANGUAGE_CODES

from genienlp.metrics import computeBLEU

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
        
        
def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    try:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    except AttributeError:
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)


def check_args(args):
    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 1)[1] in MARIAN_GROUP_MEMBERS:
        if not args.tgt_lang:
            raise ValueError('For translation task using Marian model, if target language is a group of languages, '
                             'you have to specify the --tgt_lang flag.')
        elif args.tgt_lang not in MARIAN_GROUP_MEMBERS[args.model_name_or_path.rsplit('-', 1)[1]]:
            if args.tgt_lang == 'pl':
                args.tgt_lang = 'pol'
            else:
                raise ValueError(
                    'Target language is not in the model group languages, please specify the correct target language.')
    
    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 2)[1] in MARIAN_GROUP_MEMBERS:
        if not args.src_lang:
            raise ValueError('For translation task using Marian model, if source language is a group of languages, '
                             'you have to specify the --src_lang flag.')
        elif args.src_lang not in MARIAN_GROUP_MEMBERS[args.model_name_or_path.rsplit('-', 2)[1]]:
            raise ValueError(
                'Dource language is not in the model group languages, please specify the correct source language.')
    
    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 1)[1] not in MARIAN_GROUP_MEMBERS and args.tgt_lang:
        logger.warning('Target language should not be provided when using models with single language pairs,'
                       ' otherwise the translation outputs will be incorrect; thus we ignore the target language you provided...')
        args.tgt_lang = None
    
    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 2)[1] not in MARIAN_GROUP_MEMBERS and args.src_lang:
        logger.warning('Source language should not be provided when using models with single language pairs,'
                       ' otherwise the translation outputs will be incorrect; thus we ignore the source language you provided...')
        args.src_lang = None
    
    if args.model_type == 'mbart' and not (args.tgt_lang and args.src_lang):
        raise ValueError('Source and Target language should be provided when using mBART cc25 model')

    # adjust language ids for mbart models
    if args.model_type == 'mbart':
        if args.src_lang not in FAIRSEQ_LANGUAGE_CODES:
            for lang in FAIRSEQ_LANGUAGE_CODES:
                if lang.startswith(args.src_lang):
                    args.src_lang = lang
        if args.tgt_lang not in FAIRSEQ_LANGUAGE_CODES:
            for lang in FAIRSEQ_LANGUAGE_CODES:
                if lang.startswith(args.tgt_lang):
                    args.tgt_lang = lang

def sort_checkpoints(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, "checkpointepoch=*.ckpt"), recursive=True)))


def get_transformer_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, dimension):
    num_warmup_steps = max(1, num_warmup_steps)

    def lr_lambda(current_step):
        current_step += 1
        return 1. / math.sqrt(dimension) * min(1 / math.sqrt(current_step), current_step / (num_warmup_steps * math.sqrt(num_warmup_steps)))

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

    return {'bleu': total_bleu/count, 'em': total_exact_match/count*100}


def compute_attention(sample_layer_attention, att_pooling):
    sample_layer_attention_pooled = None
    if att_pooling == 'mean':
        sample_layer_attention_pooled = torch.mean(sample_layer_attention, dim=0, keepdim=False)
    elif att_pooling == 'max':
        sample_layer_attention_pooled = torch.max(sample_layer_attention, dim=0, keepdim=False)[0]
    
    return sample_layer_attention_pooled


def replace_quoted_params(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled, model_type, tgt_lang):
    # find positions of quotation marks in src and tgt
    src2tgt_mapping = {}
    src2tgt_mapping_index = {}
    
    ## FIXED: quotation marks are exclusively used to wrap parameters so just check if they are present in target token
    # quote_wordpiece = tokenizer.tokenize('"')[0]
    # quote_token = '"'
    src_quotation_symbols = ['"']
    tgt_quotation_symbols = ['"']
    if tgt_lang == 'ru':
        tgt_quotation_symbols.extend(['«', '»'])
    
    src_spans_ind = [index for index, token in enumerate(src_tokens) if
                     any([symbol in token for symbol in src_quotation_symbols])]
    tgt_spans_ind = [index for index, token in enumerate(tgt_tokens) if
                     any([symbol in token for symbol in tgt_quotation_symbols])]
    
    if model_type == 'marian':
        src_strings = tokenizer.spm_source.DecodePieces(src_tokens)
        tgt_strings = tokenizer.spm_target.DecodePieces(tgt_tokens)
    else:
        src_strings = tokenizer.convert_tokens_to_string(src_tokens)
        tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)
    
    if len(src_spans_ind) % 2 != 0:
        logging.error('corrupted span in src string: [{}]'.format(src_strings))
        return tgt_strings, False
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
            logger.error(
                'Could not find a corresponding span in tgt for ({}, {}) src span in src string: [{}]'.format(beg, end,
                                                                                                              src_strings))
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


def force_replace_quoted_params(src_tokens, tgt_tokens, tokenizer, sample_layer_attention_pooled, model_type):
    # find positions of quotation marks in src
    src2tgt_mapping = {}
    
    src_spans_ind = [index for index, token in enumerate(src_tokens) if '"' in token]
    tgt_is_piece = [1 if token[0] == SPIECE_UNDERLINE else 0 for token in tgt_tokens]
    tgt_piece2word_mapping = list(np.cumsum(tgt_is_piece) - 1)
    
    if len(src_spans_ind) % 2 != 0:
        logging.error('corrupted span in src string: [{}]'.format(tokenizer.spm_source.DecodePieces(src_tokens)))
        # this almost never happens but if it does it is usually because quotation is missing from the end of src_tokens
        # we temporary fix this by adding '"' to the end of src_tokens
        src_tokens += tokenizer.tokenize('"')
        src_spans_ind = [index for index, token in enumerate(src_tokens) if '"' in token]
    
    if model_type == 'marian':
        src_strings = tokenizer.spm_source.DecodePieces(src_tokens)
        tgt_strings = tokenizer.spm_target.DecodePieces(tgt_tokens)
    else:
        src_strings = tokenizer.convert_tokens_to_string(src_tokens)
        tgt_strings = tokenizer.convert_tokens_to_string(tgt_tokens)
    
    # arrange spans and exclude quotation mark indices
    src_spans = [(src_spans_ind[i] + 1, src_spans_ind[i + 1] - 1) for i in range(0, len(src_spans_ind), 2)]
    
    for src_idx, (beg, end) in enumerate(src_spans):
        s1 = torch.argmax(sample_layer_attention_pooled[:, beg]).item()
        s2 = torch.argmax(sample_layer_attention_pooled[:, end]).item()
        
        # clamp values to max tgt_tokens length
        s1 = min(s1, len(tgt_tokens) - 1)
        s2 = min(s2, len(tgt_tokens) - 1)
        
        src2tgt_mapping[(beg, end)] = (s1, s2)
    
    quoted_pattern_maybe_space = re.compile(r'\"\s?([^"]*?)\s?\"')
    
    src_matches = list(re.finditer(quoted_pattern_maybe_space, src_strings))
    
    # update src2tgt_mapping to map to word indices in response
    for key, value in src2tgt_mapping.items():
        s1, s2 = value
        try:
            src2tgt_mapping[key] = (
            max(0, tgt_piece2word_mapping[s1] - 1), min(tgt_piece2word_mapping[s2] + 1, len(tgt_tokens)))
        except:
            raise ValueError('corrupted span in tgt string: [{}] with src string: [{}]\n'
                             'outputting example without reverting the parameter'.format(tgt_strings, src_strings))
    
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
        curr = end
    if curr < len(tgt_strings_words):
        tokens.extend(tgt_strings_words[curr:])
    
    text = ' '.join(tokens)
    
    return text
