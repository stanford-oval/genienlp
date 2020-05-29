from .run_pplm_discrim_train import Discriminator
from .pplm_classification_head import ClassificationHead
from .GPT2Seq2Seq import GPT2Seq2Seq
import logging
from typing import List
from transformers import GPT2LMHeadModel
import torch

class GPT2Seq2SeqWithSentiment(GPT2Seq2Seq):
    def __init__(self, config): #, class_size=3
        super().__init__(config)
        # self.pretrained_sentiment_head = ClassificationHead(
          # class_size, self.embed_size)

    def pad_to_max_length(self, input_sequences: List[List[int]]):
        """
        Adds pad tokens to the left of each input_sequence
        """
        max_length = max([len(s) for s in input_sequences])
        copy_input_sequences = []
        for i in range(len(input_sequences)):
            sep_token_index = input_sequences[i].index(self.sep_token_id)
            copy_input_sequences.append([self.pad_token_id]*(max_length-len(input_sequences[i])) + input_sequences[i])

        return copy_input_sequences

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        sep_token_position = (input_ids==self.sep_token_id).to(torch.long)
        assert (torch.sum(sep_token_position, dim=1)==1).all(), 'All input_ids must contain exactly one sep_token. sep_token_position = %s\nsep_token_id = %d' % (str(sep_token_position), self.sep_token_id)
        token_type_ids = torch.cumsum(sep_token_position, dim=1) - sep_token_position
        attention_mask = (input_ids!=self.pad_token_id).to(torch.long) # 0 means mask, 1 means no mask
        position_ids = ((torch.cumsum(attention_mask, dim=1)-1)*(1-token_type_ids)+(torch.cumsum(token_type_ids, dim=1)-1)*token_type_ids).clamp(min=0)
        token_type_ids = self.sep_token_id * (1-token_type_ids) + self.end_token_id * token_type_ids

        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        inputs = {"input_ids": input_ids, "position_ids": position_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "past": past}
        return inputs


    def forward(args):
        inputs = self.prepare_inputs_for_generation(args)
        super().forward(**inputs)

		