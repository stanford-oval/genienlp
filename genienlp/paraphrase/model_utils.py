import torch
import math
import os
import glob

from genienlp.paraphrase.data_utils import TextDataset


def get_transformer_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, dimension):
    num_warmup_steps = max(1, num_warmup_steps)

    def lr_lambda(current_step):
        current_step += 1
        return 1. / math.sqrt(dimension) * min(1 / math.sqrt(current_step), current_step / (num_warmup_steps * math.sqrt(num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def load_and_cache_examples(args, tokenizer, evaluate=False, aux=False):
    if evaluate:
        if aux:
            file_path = args.aux_eval_data_file
        else:
            file_path = args.eval_data_file
    else:
        file_path = args.train_data_file
    dataset = TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size, evaluate=evaluate)
    return dataset


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
