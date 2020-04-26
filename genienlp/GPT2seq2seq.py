from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import CrossEntropyLoss
import torch

class GPT2Seq2Seq(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.sep_token = 50258
        self.end_token = 50259
        self.pad_token = 50257

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        sep_token_position = (input_ids==self.sep_token).to(torch.long)
        assert (torch.sum(sep_token_position, dim=1)==1).all(), 'All input_ids must contain exactly one start_token'
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
    