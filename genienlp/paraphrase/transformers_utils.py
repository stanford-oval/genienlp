import re
import torch
import torch.nn.functional as F
from typing import List, Optional


from transformers import LogitsProcessorList
from transformers import MarianMTModel, BartForConditionalGeneration, MBartForConditionalGeneration,\
    T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel

from transformers.models.marian.convert_marian_to_pytorch import GROUPS
from transformers.models.marian.convert_marian_tatoeba_to_pytorch import GROUP_MEMBERS

from transformers.models.mbart.tokenization_mbart import MBartTokenizer, _all_mbart_models, SPM_URL
from transformers.models.gpt2 import tokenization_gpt2
from transformers.models.t5 import tokenization_t5

SPIECE_UNDERLINE = "▁"

language_code_re = re.compile(">>.+<<")


BART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # official models
    "facebook/bart-base": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-base/config.json",
    "facebook/bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/config.json",
    "facebook/bart-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-mnli/config.json",
    "facebook/bart-large-cnn": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/config.json",
    "facebook/bart-large-xsum": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-xsum/config.json",
    
    # community models; see https://huggingface.co/models?filter=bart for more
    "sshleifer/bart-tiny-random": "https://s3.amazonaws.com/models.huggingface.co/bert/sshleifer/bart-tiny-random/config.json"
}

MBART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # official models
    "facebook/mbart-large-en-ro": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/mbart-large-en-ro/config.json",
    "facebook/mbart-large-cc25": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/mbart-large-cc25/config.json",
    
    # community models; see https://huggingface.co/models?filter=mbart for more
    "sshleifer/tiny-mbart": "https://s3.amazonaws.com/models.huggingface.co/bert/sshleifer/tiny-mbart/config.json"
}

MT5_PRETRAINED_CONFIG_ARCHIVE_MAP = {'google/mt5-{}'.format(v): "https://s3.amazonaws.com/models.huggingface.co/bert/google/mt5-{}/config.json".format(v)
                                     for v in ['small', 'base', 'large', 'xl', 'xxl']}


BART_MODEL_LIST = list(BART_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
MBART_MODEL_LIST = list(MBART_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
MT5_MODEL_LIST = list(MT5_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())


# all MarianMT models use the same config
MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Helsinki-NLP/opus-mt-en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP/opus-mt-en-de/config.json",
}


MARIAN_GROUPS = {item[1]: set(item[0].split('+')) for item in GROUPS}
MARIAN_TATOEBA_GROUPS = {k: set(v[1]) for k, v in GROUP_MEMBERS.items()}

MARIAN_GROUP_MEMBERS = {**MARIAN_GROUPS, **MARIAN_TATOEBA_GROUPS}


MODEL_PARALLEL_SUPPORTED_MODELS = set(list(tokenization_gpt2.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES.keys()) +
                                       list(tokenization_t5.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES.keys()) +
                                       list(MT5_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()))

###############

class GenieMBartTokenizer(MBartTokenizer):
    '''
    MBartTokenizer with the temporary fix for off-by-one error during generation: https://github.com/huggingface/transformers/issues/5755
    '''
    vocab_files_names = {"vocab_file": "sentencepiece.bpe.model"}
    max_model_input_sizes = {m: 1024 for m in _all_mbart_models}
    pretrained_vocab_files_map = {"vocab_file": {m: SPM_URL for m in _all_mbart_models}}

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(self, *args, tokenizer_file=None, **kwargs):
        super().__init__(*args, tokenizer_file=tokenizer_file, **kwargs)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. Prefix [bos_token_id], suffix =[eos_token_id]."""
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        self.prefix_tokens = [self.bos_token_id]
        self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. Prefix [tgt_lang_code], suffix =[eos_token_id]."""
        self.cur_lang_code = self.lang_code_to_id[lang]
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]
        
###############


class GeniePreTrainedModel(PreTrainedModel):
    '''
    General class for PreTrainedModel which can output cross-attention weights during generation
    '''
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
    
    def greedy_search(
            self,
            input_ids: torch.LongTensor,
            logits_processor=None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            **model_kwargs
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        
        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )
        
        output_attentions = model_kwargs.get('output_attentions', None)
        
        if output_attentions:
            batch_size = input_ids.size(0)
            if getattr(self.config, 'encoder_layers', None):
                num_layers = self.config.encoder_layers
            else:
                num_layers = self.config.num_layers
            
            if getattr(self.config, 'encoder_attention_heads', None):
                num_heads = self.config.encoder_attention_heads
            else:
                num_heads = self.config.num_heads
            
            if model_kwargs.get('encoder_outputs', None):
                seq_length = model_kwargs['encoder_outputs'][0].size(1)
            else:
                seq_length = max_length
                
            all_cross_attentions = [input_ids.new_full([batch_size, num_heads, max_length, seq_length],
                                                       dtype=torch.float32,
                                                       fill_value=-1000000)
                                    for _ in range(num_layers)]
   
        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            
            if output_attentions:
                for i in range(num_layers):
                    all_cross_attentions[i][:, :, [cur_len - 1], :] = outputs.cross_attentions[i]
            
            # pre-process distribution
            scores = logits_processor(input_ids, next_token_logits)
            
            # argmax
            next_tokens = torch.argmax(scores, dim=-1)
            
            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)
            
            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )
            
            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            
            # stop when there is a </s> in each sentence, or if we exceed the maximum length
            if unfinished_sequences.max() == 0:
                break
            
            # increase cur_len
            cur_len = cur_len + 1
        
        if output_attentions:
            # List of each encoder layer cross-attention values each with size (bsz, num_heads, tgt_len, src_len)
            all_cross_attentions = [layer_all_cross_attentions[:, :, :sequence_lengths.max().item(), :] for
                                    layer_all_cross_attentions in all_cross_attentions]
            
            return input_ids, all_cross_attentions
        else:
            return input_ids
    
    
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor=None,
        logits_warper=None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **model_kwargs
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self._init_sequence_length_for_generation(
            input_ids, max_length
        )
        
        output_attentions = model_kwargs.get('output_attentions', None)
        
        if output_attentions:
            batch_size = input_ids.size(0)
            if getattr(self.config, 'encoder_layers', None):
                num_layers = self.config.encoder_layers
            else:
                num_layers = self.config.num_layers
    
            if getattr(self.config, 'encoder_attention_heads', None):
                num_heads = self.config.encoder_attention_heads
            else:
                num_heads = self.config.num_heads
    
            if model_kwargs.get('encoder_outputs', None):
                seq_length = model_kwargs['encoder_outputs'][0].size(1)
            else:
                seq_length = max_length
    
            all_cross_attentions = [input_ids.new_full([batch_size, num_heads, max_length, seq_length],
                                                       dtype=torch.float32,
                                                       fill_value=-1000000)
                                    for _ in range(num_layers)]

        # auto-regressive generation
        while cur_len < max_length:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            
            if output_attentions:
                for i in range(num_layers):
                    all_cross_attentions[i][:, :, [cur_len - 1], :] = outputs.cross_attentions[i]

            # pre-process distribution
            scores = logits_processor(input_ids, next_token_logits)
            scores = logits_warper(input_ids, scores)

            # sample
            probs = F.softmax(scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sequences.max() == 0:
                break

            # update model kwargs
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
        
        if output_attentions:
            # List of each encoder layer cross-attention values each with size (bsz, num_heads, tgt_len, src_len)
            all_cross_attentions = [layer_all_cross_attentions[:, :, :sequence_lengths.max().item(), :] for
                                    layer_all_cross_attentions in all_cross_attentions]
    
            return input_ids, all_cross_attentions
        else:
            return input_ids

###############

class GenieMarianMTModel(MarianMTModel, GeniePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

class GenieBartForConditionalGeneration(BartForConditionalGeneration, GeniePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
class GenieMBartForConditionalGeneration(MBartForConditionalGeneration, GeniePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

class GenieT5ForConditionalGeneration(T5ForConditionalGeneration, GeniePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

class GenieMT5ForConditionalGeneration(MT5ForConditionalGeneration, GeniePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
