import torch
import math
import os
import glob
import re
import logging
import shutil

from genienlp.metrics import computeBLEU

logger = logging.getLogger(__name__)

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