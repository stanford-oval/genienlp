import torch
import math
import os
import glob
import re
import logging
import shutil

from .transformers_utils import MARIAN_GROUP_MEMBERS
from transformers.models.mbart.tokenization_mbart50 import FAIRSEQ_LANGUAGE_CODES

from genienlp.metrics import computeBLEU
from genienlp.util import get_mbart_lang

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
            raise ValueError('For translation task using Marian model, if target language is a group of languages, '
                             'you have to specify the --tgt_lang flag.')
        elif args.tgt_lang not in MARIAN_GROUP_MEMBERS[args.model_name_or_path.rsplit('-', 1)[1]]:
            if args.tgt_lang == 'pl':
                args.tgt_lang = 'pol'
            elif args.tgt_lang == 'fa':
                args.tgt_lang = 'pes'
            else:
                raise ValueError(
                    'Target language is not in the model group languages, please specify the correct target language.')
    
    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 2)[1] in MARIAN_GROUP_MEMBERS:
        if not args.src_lang:
            raise ValueError('For translation task using Marian model, if source language is a group of languages, '
                             'you have to specify the --src_lang flag.')
        elif args.src_lang not in MARIAN_GROUP_MEMBERS[args.model_name_or_path.rsplit('-', 2)[1]]:
            if args.src_lang == 'pl':
                args.src_lang = 'pol'
            elif args.src_lang == 'fa':
                args.src_lang = 'pes'
            raise ValueError(
                'Source language is not in the model group languages, please specify the correct source language.')
    
    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 1)[1] not in MARIAN_GROUP_MEMBERS and args.tgt_lang:
        logger.warning('Target language should not be provided when using Marian models with single target language,'
                       ' otherwise the translation outputs will be incorrect; thus we ignore the target language you provided...')
        args.tgt_lang = None
    
    if args.model_type == 'marian' and args.model_name_or_path.rsplit('-', 2)[1] not in MARIAN_GROUP_MEMBERS and args.src_lang:
        logger.warning('Source language should not be provided when using Marian models with single source language,'
                       ' otherwise the translation outputs will be incorrect; thus we ignore the source language you provided...')
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


