import os
import glob

import torch
from torch.utils.data import Dataset

from transformers.tokenization_utils import trim_batch

def sort_checkpoints(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, "checkpointepoch=*.ckpt"), recursive=True)))


def mask_tokens(inputs, labels, tokenizer, args):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    # print('labels.tolist() = ', labels.tolist())
    # print('special_tokens_mask = ', special_tokens_mask)
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def remove_thingtalk_quotes(thingtalk):
    while True:
        l1 = thingtalk.find('"')
        if l1 < 0:
            break
        l2 = thingtalk.find('"', l1+1)
        assert l2 >= 0
        thingtalk = thingtalk[:l1] + '<temp>' + thingtalk[l2+1:]
    return thingtalk

def get_inbetween_tokens(train_data_file, start_token, end_token):
    new_tokens = set()
    with open(train_data_file, encoding="utf-8") as f:
        for line in tqdm(f, desc='Tokenizing'):
            line = remove_thingtalk_quotes(line)
            line = line[line.find(start_token)+len(start_token):line.find(end_token)]
            new_tokens.update(line.split())
    return new_tokens

def add_special_tokens(model, tokenizer, additional_special_tokens, pad_token=None):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': additional_special_tokens}
    if pad_token is not None:
        ATTR_TO_SPECIAL_TOKEN['pad_token'] = pad_token
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        logger.info('Added %d special tokens', num_added_tokens)
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path,
        max_source_length,
        max_target_length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = os.path.join(data_dir, type_path + ".tsv")
        self.source = self.encode_file(max_source_length, column=0)
        self.target = self.encode_file(max_target_length, column=1)

    def __len__(self):
        return self.source["input_ids"].shape[0]

    def __getitem__(self, index):
        source_ids = self.source["input_ids"][index].squeeze()
        target_ids = self.target["input_ids"][index].squeeze()
        src_mask = self.source["attention_mask"][index].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        target = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, target
    
    def encode_file(self, max_length, column, pad_to_max_length=True, return_tensors="pt"):
        values = []
        with open(self.data_path, "r") as f:
            for line in f:
                line = tuple(map(lambda part: part.strip(), line.split('\t')))[column]
                values.append(line)
        encoded_values = self.tokenizer.batch_encode_plus(values, max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors)
        return encoded_values

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        target = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": target}

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, block_size=512, evaluate=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, os.path.basename(os.path.normpath(args.model_name_or_path)) + '_cached_lm_' + str(self.block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples, self.labels, self.position_ids, self.segment_ids = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", file_path)

            self.prompt_token_id = self.tokenizer.convert_tokens_to_ids(args.start_special_token)
            self.end_token_id = self.tokenizer.convert_tokens_to_ids(args.end_special_token)
            self.segment1_id = 0
            self.segment2_id = 1
            if args.model_type == 'gpt2':
                self.segment1_id = self.prompt_token_id
                self.segment2_id = self.end_token_id
            # print('prompt_token_id = ', prompt_token_id)
            self.examples = []
            self.labels = []
            self.position_ids = []
            self.segment_ids = []
            self.max_input_length = 0

            if not evaluate and args.aux_train_data_file is not None:
                number_of_lines = get_number_of_lines(args.aux_train_data_file)
                with open(args.aux_train_data_file, encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in tqdm(reader, desc='Tokenizing Auxiliary File', total=number_of_lines):
                        self._add_example(row[0], None, args)

            number_of_lines = get_number_of_lines(file_path)
            with open(file_path, encoding="utf-8") as f:
                reader = csv.reader(f, delimiter='\t')
                for row in tqdm(reader, desc='Tokenizing', total=number_of_lines):
                    self._add_example(row[0], row[1], args)

            

            logger.info('Maximum input length: %d', self.max_input_length)
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump((self.examples, self.labels, self.position_ids, self.segment_ids), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _add_example(self, input_sequence, output_sequence, args):
        """
        Args:
            input_sequence: if None, a corrupted version of the output_sequence will be used
        """
        # TODO we should make use of tokenizer.build_inputs_with_special_tokens(sequence1, sequence2). Add special tokens manualy only if our model does not support two sequences (like GPT2).
        
        input_token_ids = self.tokenizer.encode(input_sequence, add_special_tokens=False) + [self.tokenizer.convert_tokens_to_ids(args.start_special_token)]
        if output_sequence is None:
            output_token_ids = []
        else:
            output_token_ids = self.tokenizer.encode(output_sequence, add_special_tokens=False) + [self.tokenizer.convert_tokens_to_ids(args.end_special_token)]
        tokenized_text = input_token_ids + output_token_ids
        
        tokenized_text = tokenized_text[0:self.block_size] # truncate longer sequences
        # print('tokenized_text = ', tokenized_text)

        example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
        # Remove duplicate end_token for models like BERT and RoBERTa that already add it
        if example[-2] == self.end_token_id:
            example = example[:-1]
        # print('example = ', example)
        self.max_input_length = max(self.max_input_length, len(example))
        try:
            prompt_token_location = example.index(self.prompt_token_id)
        except ValueError:
            logger.warning('Prompt token not found after truncating the input. Dropping the example.')
            return

        self.examples.append(example)
        if args.train_all_tokens and not evaluate or output_sequence is None:
            self.labels.append(example)
        else: # During evaluation, we only care about the output_sequence so we mask the input
            self.labels.append([-100]*(prompt_token_location+1)+example[prompt_token_location+1:])
        
        position_ids2 = range(len(example)-prompt_token_location-1)
        if args.reverse_position_ids:
            position_ids2 = reversed(position_ids2)
        self.position_ids.append(list(range(prompt_token_location+1)) + list(position_ids2))
        self.segment_ids.append([self.segment1_id]*(prompt_token_location+1) + [self.segment2_id]*(len(example)-prompt_token_location-1))

        # print('position_ids = ', self.position_ids[-1])
        # print('segment_ids = ', self.segment_ids[-1])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.labels[item]), torch.tensor(self.position_ids[item]), torch.tensor(self.segment_ids[item])


    def collate_fn(self, batch, pad_token_id):
        (inputs, labels, position_ids, segment_ids) = zip(*batch)
        inputs_pad = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100)
        position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0) # will be ignored in the loss function, so its value does not matter
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0) # will be ignored in the loss function, so its value does not matter
    
        return inputs_pad, labels_pad, position_ids, segment_ids
