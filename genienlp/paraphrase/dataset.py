import logging
import os
import pickle
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..data_utils.almond_utils import detokenize_cjk_chars
from ..data_utils.progbar import progress_bar
from ..util import get_number_of_lines

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, block_size=512, evaluate=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            os.path.basename(os.path.normpath(args.model_name_or_path))
            + '_cached_lm_'
            + str(self.block_size)
            + '_'
            + filename,
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.input_ids, self.labels, self.position_ids, self.segment_ids = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", file_path)

            self.prompt_token_id = self.tokenizer.convert_tokens_to_ids(args.start_special_token)
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids(args.end_special_token)
            self.segment1_id = 0
            self.segment2_id = 1
            if args.model_type == 'gpt2':
                self.segment1_id = self.prompt_token_id
                self.segment2_id = self.eos_token_id
            self.input_ids = []
            self.attention_mask = []
            self.labels = []
            self.position_ids = []
            self.segment_ids = []
            self.max_input_length = 0
            self.max_output_length = 0
            self.evaluate = evaluate

            if not self.evaluate and args.aux_train_data_file is not None:
                number_of_lines = get_number_of_lines(args.aux_train_data_file)
                lines = min(args.subsample, number_of_lines)
                i = 0
                with open(args.aux_train_data_file, encoding="utf-8") as f_in:
                    for line in progress_bar(f_in, desc='Tokenizing Auxiliary File', total=number_of_lines):
                        parts = list(map(lambda part: part.strip(), line.split('\t')))
                        i += 1
                        if i > lines:
                            break

                        parts[args.input_column] = args.model_input_prefix + parts[args.input_column]
                        if args.model_type in ['bart', 'mbart', 'marian']:
                            self._add_seq2seq_example(parts[args.input_column], None, args)
                        else:
                            self._add_example(parts[args.input_column], None, args)

            number_of_lines = get_number_of_lines(file_path)
            lines = min(args.subsample, number_of_lines)
            i = 0
            with open(file_path, encoding="utf-8") as f_in:
                for line in progress_bar(f_in, desc='Tokenizing', total=number_of_lines):
                    parts = list(map(lambda part: part.strip(), line.split('\t')))
                    i += 1
                    if i > lines:
                        break

                    parts[args.input_column] = args.model_input_prefix + parts[args.input_column]
                    if args.model_type in ['bart', 'mbart', 'marian']:
                        self._add_seq2seq_example(parts[args.input_column], parts[args.gold_column], args)
                    else:
                        self._add_example(parts[args.input_column], parts[args.gold_column], args)
            if args.sort_by_length:
                _, self.input_ids, self.attention_mask, self.labels, self.position_ids, self.segment_ids = tuple(
                    zip(
                        *sorted(
                            list(
                                zip(
                                    [len(x) for x in self.input_ids],
                                    self.input_ids,
                                    self.attention_mask,
                                    self.labels,
                                    self.position_ids,
                                    self.segment_ids,
                                )
                            )
                        )
                    )
                )
            logger.info('Maximum input length: %d', self.max_input_length)

            if args.cache_input_data:
                logger.info("Saving features into cached file %s", cached_features_file)
                with open(cached_features_file, 'wb') as handle:
                    pickle.dump(
                        (self.input_ids, self.labels, self.position_ids, self.segment_ids),
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

    def _add_example(self, input_sequence, output_sequence, args):
        # TODO we should make use of tokenizer.build_inputs_with_special_tokens(sequence1, sequence2). Add special tokens manualy only if our model does not support two sequences (like GPT2).

        input_sequence = detokenize_cjk_chars(input_sequence)
        if output_sequence is not None:
            output_sequence = detokenize_cjk_chars(output_sequence)

        input_token_ids = self.tokenizer.encode(input_sequence, add_special_tokens=False) + [
            self.tokenizer.convert_tokens_to_ids(args.start_special_token)
        ]
        if output_sequence is None:
            output_token_ids = []
        else:
            output_token_ids = self.tokenizer.encode(output_sequence, add_special_tokens=False) + [
                self.tokenizer.convert_tokens_to_ids(args.end_special_token)
            ]
        tokenized_text = input_token_ids + output_token_ids

        # do not use exampels that are too long for the model (for supervised tasks, this is better than truncating examples)
        if len(tokenized_text) > self.block_size:
            logger.warning(
                'Skipping example with length %d which was longer than block size (%d)', len(tokenized_text), self.block_size
            )
            return

        input_ids = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
        # Remove duplicate end_token for models like BERT and RoBERTa that already add it
        if input_ids[-2] == self.eos_token_id:
            input_ids = input_ids[:-1]
        self.max_input_length = max(self.max_input_length, len(input_ids))
        try:
            prompt_token_location = input_ids.index(self.prompt_token_id)
        except ValueError:
            logger.warning('Prompt token not found after truncating the input. Dropping the example.')
            return

        self.input_ids.append(input_ids)
        if args.train_all_tokens and not self.evaluate or output_sequence is None:
            self.labels.append(input_ids)
        else:  # During evaluation, we only care about the output_sequence so we mask the input
            self.labels.append([args.mlm_ignore_index] * (prompt_token_location + 1) + input_ids[prompt_token_location + 1 :])

        position_ids2 = range(len(input_ids) - prompt_token_location - 1)
        if args.reverse_position_ids:
            position_ids2 = reversed(position_ids2)
        self.position_ids.append(list(range(prompt_token_location + 1)) + list(position_ids2))
        self.segment_ids.append(
            [self.segment1_id] * (prompt_token_location + 1)
            + [self.segment2_id] * (len(input_ids) - prompt_token_location - 1)
        )

        # ignored
        self.attention_mask.append([1] * len(input_ids))

    def _add_seq2seq_example(self, input_sequence, output_sequence, args):

        input_sequence = detokenize_cjk_chars(input_sequence)
        if output_sequence is not None:
            output_sequence = detokenize_cjk_chars(output_sequence)

        if args.model_type == 'mbart':
            model_inputs = self.tokenizer.prepare_seq2seq_batch(
                [input_sequence], args.src_lang, [output_sequence], args.tgt_lang
            )
        else:
            model_inputs = self.tokenizer.prepare_seq2seq_batch([input_sequence], [output_sequence], return_tensors='pt')

        encoded_input_ids = model_inputs['input_ids'].tolist()[0]
        encoded_attention_mask = model_inputs['attention_mask'].tolist()[0]
        encoded_output_ids = model_inputs['labels'].tolist()[0]

        self.max_input_length = max(self.max_input_length, len(encoded_input_ids))
        self.max_output_length = max(self.max_output_length, len(encoded_output_ids))

        self.input_ids.append(encoded_input_ids)
        self.attention_mask.append(encoded_attention_mask)
        self.position_ids.append(list(range(len(encoded_input_ids))))
        self.segment_ids.append([self.segment1_id] * len(encoded_input_ids))

        self.labels.append(encoded_output_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return (
            torch.tensor(self.input_ids[item]),
            torch.tensor(self.attention_mask[item]),
            torch.tensor(self.labels[item]),
            torch.tensor(self.position_ids[item]),
            torch.tensor(self.segment_ids[item]),
        )

    def collate_fn(self, batch):
        (inputs, attention_mask, labels, position_ids, segment_ids) = zip(*batch)
        inputs_pad = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        position_ids = pad_sequence(
            position_ids, batch_first=True, padding_value=0
        )  # will be ignored in the loss function, so its value does not matter
        segment_ids = pad_sequence(
            segment_ids, batch_first=True, padding_value=0
        )  # will be ignored in the loss function, so its value does not matter

        return inputs_pad, attention_mask, labels_pad, position_ids, segment_ids


class LengthSortedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, shuffle):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_returned_items = 0
        self.last_batch_start_index = 0
        self.last_batch_start_index = self._get_next_batch_start_index()
        self.last_batch_size = 0

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        self.total_returned_items = 0
        self.last_batch_start_index = 0
        self.last_batch_start_index = self._get_next_batch_start_index()
        self.last_batch_size = 0
        return self

    def __next__(self):
        ret = (self.last_batch_start_index + self.last_batch_size) % len(self)
        self.total_returned_items = self.total_returned_items + 1
        if self.total_returned_items > len(self):
            raise StopIteration
        self.last_batch_size = self.last_batch_size + 1
        if self.last_batch_size == self.batch_size:
            self.last_batch_size = 0
            self.last_batch_start_index = self._get_next_batch_start_index()

        return ret

    def _get_next_batch_start_index(self):
        if self.shuffle:
            return random.randint(0, len(self))
        else:
            return self.last_batch_start_index + self.batch_size
