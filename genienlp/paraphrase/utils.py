import os

import torch
from torch.utils.data import Dataset
import csv

from transformers.tokenization_utils import trim_batch


def encode_file(tokenizer, data_path, max_length, pad_to_max_length=True, return_tensors="pt"):
    sources = []
    targets = []
    with open(data_path, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            sources.append(row[0])
            targets.append(row[1])
        encoded_source = tokenizer.batch_encode_plus(sources, max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors)
        encoded_target = tokenizer.batch_encode_plus(targets, max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors)
    return encoded_source, encoded_target


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path,
        max_length,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.source, self.target = encode_file(tokenizer, os.path.join(data_dir, type_path + ".tsv"), max_length)

    def __len__(self):
        return self.source["input_ids"].shape[0]

    def __getitem__(self, index):
        source_ids = self.source["input_ids"][index].squeeze()
        target_ids = self.target["input_ids"][index].squeeze()
        src_mask = self.source["attention_mask"][index].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}
