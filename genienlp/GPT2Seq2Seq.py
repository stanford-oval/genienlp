from typing import List
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import torch

class GPT2Seq2Seq(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.end_token = 50259
        self.sep_token = 50258
        self.pad_token = 50257


    def pad_to_max_length(self, input_sequences: List[List[int]]):
        """
        Adds pad tokens before the sep_token
        """
        max_length = len(input_sequences[0]) # input is sorted by length
        copy_input_sequences = []
        for i in range(len(input_sequences)):
            sep_token_index = input_sequences[i].index(self.sep_token)
            copy_input_sequences.append(input_sequences[i][:sep_token_index] + \
                                        [self.pad_token]*(max_length-len(input_sequences[i])) +\
                                        input_sequences[i][sep_token_index:])
        
        return copy_input_sequences

    
    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """ repetition penalty from CTRL (https://arxiv.org/abs/1909.05858), but much faster on GPU
        """
        if repetition_penalty == 1.0:
            return lprobs
        m = torch.scatter(input=torch.zeros_like(lprobs), dim=1, index=prev_output_tokens, value=1)
        m[:self.sep_token] = 0
        m[:self.pad_token] = 0
        # logger.info('m = ', m.shape)
        need_change = m * lprobs
        need_divide = need_change > 0
        need_multiply = need_change < 0
        lprobs = need_divide * lprobs / repetition_penalty + need_multiply * lprobs * repetition_penalty + (1-m) * lprobs
        
        # old, slow implementation
        # if repetition_penalty != 1.0:
            # for i in range(context.shape[0]):
                # for previous_token in set(generated[i].tolist()):
                    # if lprobs[i, previous_token] > 0:
                        # lprobs[i, previous_token] /= repetition_penalty
                    # else:
                        # lprobs[i, previous_token] *= repetition_penalty


    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        sep_token_position = (input_ids==self.sep_token).to(torch.long)
        assert (torch.sum(sep_token_position, dim=1)==1).all(), 'All input_ids must contain exactly one start_token. sep_token_position = %s' % str(sep_token_position)
        token_type_ids = torch.cumsum(sep_token_position, dim=1) - sep_token_position
        attention_mask = (input_ids!=self.pad_token).to(torch.long) # 0 means mask, 1 means no mask
        position_ids = (torch.cumsum(attention_mask, dim=1)-1)*(1-token_type_ids)+(torch.cumsum(token_type_ids, dim=1)-1)*token_type_ids
        token_type_ids = self.sep_token * (1-token_type_ids) + self.end_token * token_type_ids
        # print('input_ids = ', input_ids)
        # print('position_ids = ', position_ids)
        # print('token_type_ids = ', token_type_ids)
        # print('attention_mask = ', attention_mask)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            attention_mask = attention_mask[:, -1].unsqueeze(-1)

        inputs = {"input_ids": input_ids, "position_ids": position_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "past": past}
        return inputs

if __name__ == '__main__':
    model = GPT2Seq2Seq.from_pretrained('workdir/models/gpt2-medium-5')
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('workdir/models/gpt2-medium-5')
    # print(tokenizer.convert_tokens_to_ids('</paraphrase>'))
    # print(tokenizer.convert_tokens_to_ids('<paraphrase>'))
    dct = tokenizer.batch_encode_plus(['show me restaurants around here. <paraphrase>', 'where is it? <paraphrase>'], return_tensors="pt", pad_to_max_length=True)
    outputs = model.generate(input_ids=dct['input_ids'],
                            max_length=40,
                            num_beams=16,
                            early_stopping=True,
                            num_return_sequences=4,
                            do_sample=False,
                            temperature=1.0,
                            eos_token_id=50259,
                            pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))  # do greedy decoding
    print('outputs = ', outputs)
    for output in outputs:
        print('Generated: {}'.format(tokenizer.decode(output, skip_special_tokens=True)))
    