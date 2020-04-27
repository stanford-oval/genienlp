import argparse
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from genienlp.paraphrase.finetune_bart import MBART
from genienlp.paraphrase.utils import Seq2SeqDataset


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def generate_summaries(args, mbart, tokenizer, dataloader):
    device = args.device

    max_length = 140
    min_length = 1
    
    f_out = open(args.output_path, 'w')
    for batch in tqdm(dataloader):
        summaries = mbart.model.generate(
            input_ids=batch["source_ids"].to(device),
            attention_mask=batch["source_mask"].to(device),
            num_beams=1,
            do_sample=False,
            temperature=1,
            length_penalty=1,
            max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=min_length + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=mbart.config.eos_token_id,
            num_return_sequences=1
        )
        results = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summaries]
        for hypothesis in results:
            f_out.write(hypothesis + "\n")
            f_out.flush()
    f_out.close()


def run_generate(args):

    
    mbart = MBART.load_from_checkpoint(os.path.join(args.ckpt_path, args.ckpt_name))
    mbart.eval()
    mbart = mbart.to(args.device)
    tokenizer = mbart.tokenizer.from_pretrained(args.model_name)
    dataset = Seq2SeqDataset(tokenizer, args.source_path, args.predict_split, args.max_source_length, args.max_target_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, num_workers=1, shuffle=False)
    generate_summaries(args, mbart, tokenizer, dataloader)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, help="like cnn_dm/; should contain ending slash" )
    parser.add_argument("--output_path", type=str, help="where to save summaries")
    parser.add_argument("--model_name", type=str, default="bart-large-cnn", help="like bart-large-cnn")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--batch_size", type=int, default=8, required=False, help="batch size: how many to summarize at a time")
    parser.add_argument("--ckpt_path", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--ckpt_name", type=str, default='checkpointepoch=0.ckpt', help="checkpoint name")
    parser.add_argument("--predict_split", type=str, default='test', help="data split to run prediction for")
    parser.add_argument(
        "--max_source_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=56,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    args = parser.parse_args()
    
    run_generate(args)
