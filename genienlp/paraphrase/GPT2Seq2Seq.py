from typing import List

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


class GPT2Seq2Seq(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def set_token_ids(self, eos_token_id, sep_token_id, pad_token_id):
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id

    def pad_to_max_length(self, input_sequences: List[List[int]]):
        """
        Adds pad tokens to the left of each input_sequence
        """
        max_length = max([len(s) for s in input_sequences])
        copy_input_sequences = []
        for i in range(len(input_sequences)):
            copy_input_sequences.append([self.pad_token_id] * (max_length - len(input_sequences[i])) + input_sequences[i])

        return copy_input_sequences

    # TODO check if this function is used
    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """repetition penalty from CTRL (https://arxiv.org/abs/1909.05858), but much faster on GPU"""
        if repetition_penalty == 1.0:
            return lprobs
        m = torch.scatter(input=torch.zeros_like(lprobs), dim=1, index=prev_output_tokens, value=1)
        m[: self.sep_token_id] = 0
        m[: self.pad_token_id] = 0
        # logger.info('m = ', m.shape)
        need_change = m * lprobs
        need_divide = need_change > 0
        need_multiply = need_change < 0
        lprobs = need_divide * lprobs / repetition_penalty + need_multiply * lprobs * repetition_penalty + (1 - m) * lprobs

    def generate(self, **kwargs):
        # change arguments so that they have the same meaning as seq2seq models
        if kwargs['bad_words_ids'] is None:
            kwargs['bad_words_ids'] = []
        kwargs['bad_words_ids'].append([self.sep_token_id])
        if kwargs['min_length'] is None:
            kwargs['min_length'] = 0
        kwargs['min_length'] += kwargs['input_ids'].shape[1]
        if kwargs['max_length'] is None:
            kwargs['max_length'] = 0
        kwargs['max_length'] += kwargs['input_ids'].shape[1]

        outputs = super().generate(**kwargs)
        sequences = outputs.sequences
        sequences = sequences[:, :].tolist()
        for i in range(len(outputs)):
            sequences[i] = [x for x in sequences[i] if x != self.pad_token_id]  # remove padding
            sequences[i] = sequences[i][
                sequences[i].index(self.sep_token_id) + 1 :
            ]  # only return the output (i.e. after sep_token)
        outputs.sequences = sequences

        return outputs

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        sep_token_position = (input_ids == self.sep_token_id).to(torch.long)
        assert (
            torch.sum(sep_token_position, dim=1) == 1
        ).all(), 'All input_ids must contain exactly one sep_token.' ' sep_token_position = %s\nsep_token_id = %d' % (
            str(sep_token_position),
            self.sep_token_id,
        )
        token_type_ids = torch.cumsum(sep_token_position, dim=1) - sep_token_position
        attention_mask = (input_ids != self.pad_token_id).to(torch.long)  # 0 means mask, 1 means no mask
        position_ids = (
            (torch.cumsum(attention_mask, dim=1) - 1) * (1 - token_type_ids)
            + (torch.cumsum(token_type_ids, dim=1) - 1) * token_type_ids
        ).clamp(min=0)
        token_type_ids = self.sep_token_id * (1 - token_type_ids) + self.eos_token_id * token_type_ids

        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
        }
        return inputs
