import os
import torch
import pickle
import logging
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from genienlp.util import get_number_of_lines

logger = logging.getLogger(__name__)


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
            self.labels = []
            self.position_ids = []
            self.segment_ids = []
            self.max_input_length = 0
            self.max_output_length = 0
            self.evaluate = evaluate

            if not self.evaluate and args.aux_train_data_file is not None:
                number_of_lines = get_number_of_lines(args.aux_train_data_file)
                with open(args.aux_train_data_file, encoding="utf-8") as f_in:
                    for line in tqdm(f_in, desc='Tokenizing Auxiliary File', total=number_of_lines):
                        parts = list(map(lambda part: part.strip(), line.split('\t')))
                    if 'bart' in args.model_type:
                        self._add_bart_example(parts[0], None, args)
                    else:
                        self._add_example(parts[0], None, args)

            number_of_lines = get_number_of_lines(file_path)
            with open(file_path, encoding="utf-8") as f_in:
                for line in tqdm(f_in, desc='Tokenizing', total=number_of_lines):
                    parts = list(map(lambda part: part.strip(), line.split('\t')))
                    if 'bart' in args.model_type:
                        self._add_bart_example(parts[0], parts[1], args)
                    else:
                        self._add_example(parts[0], parts[1], args)

            
            logger.info('Maximum input length: %d', self.max_input_length)
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump((self.input_ids, self.labels, self.position_ids, self.segment_ids), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _add_example(self, input_sequence, output_sequence, args):
        # TODO we should make use of tokenizer.build_inputs_with_special_tokens(sequence1, sequence2). Add special tokens manualy only if our model does not support two sequences (like GPT2).
        
        input_token_ids = self.tokenizer.encode(input_sequence, add_special_tokens=False) + [self.tokenizer.convert_tokens_to_ids(args.start_special_token)]
        if output_sequence is None:
            output_token_ids = []
        else:
            output_token_ids = self.tokenizer.encode(output_sequence, add_special_tokens=False) + [self.tokenizer.convert_tokens_to_ids(args.end_special_token)]
        tokenized_text = input_token_ids + output_token_ids
        
        tokenized_text = tokenized_text[0:self.block_size] # truncate longer sequences

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
        else: # During evaluation, we only care about the output_sequence so we mask the input
            self.labels.append([args.mlm_ignore_index]*(prompt_token_location+1)+input_ids[prompt_token_location+1:])
        
        position_ids2 = range(len(input_ids)-prompt_token_location-1)
        if args.reverse_position_ids:
            position_ids2 = reversed(position_ids2)
        self.position_ids.append(list(range(prompt_token_location+1)) + list(position_ids2))
        self.segment_ids.append([self.segment1_id]*(prompt_token_location+1) + [self.segment2_id]*(len(input_ids)-prompt_token_location-1))


    def _add_bart_example(self, input_sequence, output_sequence, args):
        # TODO we should make use of tokenizer.build_inputs_with_special_tokens(sequence1, sequence2). Add special tokens manualy only if our model does not support two sequences (like GPT2).
        
        encoded_input_ids = self.tokenizer.encode_plus(input_sequence)['input_ids']
        encoded_output_ids = self.tokenizer.encode_plus(output_sequence)['input_ids']
        
        self.max_input_length = max(self.max_input_length, len(encoded_input_ids))
        self.max_output_length = max(self.max_output_length, len(encoded_output_ids))
        
        self.input_ids.append(encoded_input_ids)
        self.position_ids.append(list(range(len(encoded_input_ids))))
        self.segment_ids.append([self.segment1_id] * len(encoded_input_ids))
        
        self.labels.append(encoded_output_ids)
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return torch.tensor(self.input_ids[item]), torch.tensor(self.labels[item]), torch.tensor(self.position_ids[item]), torch.tensor(self.segment_ids[item])


    def collate_fn(self, batch):
        (inputs, labels, position_ids, segment_ids) = zip(*batch)
        inputs_pad = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100)
        position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0) # will be ignored in the loss function, so its value does not matter
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0) # will be ignored in the loss function, so its value does not matter
    
        return inputs_pad, labels_pad, position_ids, segment_ids
