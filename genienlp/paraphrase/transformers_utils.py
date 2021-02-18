import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

from transformers import XLMRobertaConfig

from transformers import LogitsProcessorList
from transformers import MarianMTModel, BartForConditionalGeneration, MBartForConditionalGeneration,\
    T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel

from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids, RobertaEncoder, \
    RobertaPooler, RobertaPreTrainedModel

from transformers.models.marian.convert_marian_to_pytorch import GROUPS
from transformers.models.marian.convert_marian_tatoeba_to_pytorch import GROUP_MEMBERS
from transformers.models.marian.tokenization_marian import MarianTokenizer

from transformers.models.mbart.tokenization_mbart import MBartTokenizer, _all_mbart_models, SPM_URL
from transformers.models.gpt2 import tokenization_gpt2
from transformers.models.t5 import tokenization_t5

SPIECE_UNDERLINE = "▁"

logger = logging.getLogger(__name__)

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


MODEL_PARALLEL_SUPPORTED_MODELS = list(tokenization_gpt2.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES.keys()) + \
                                    list(tokenization_t5.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES.keys()) + \
                                    list(MT5_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())

###############


class GenieMBartTokenizer(MBartTokenizer):
    '''
    MBartTokenizer with the temporary fix for off-by-one error during generation: https://github.com/huggingface/transformers/issues/5755
    '''

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


class GenieMarianTokenizer(MarianTokenizer):
    '''
    MarianTokenizer with the temporary fix for decoding.
    In current huggingface's implementation `convert_tokens_to_string` method always uses spm_target to decode.
    To be able to decode both source language and target language, this class changes the spm to be current_spm
    '''
    
    def __init__(self, vocab, source_spm, target_spm, source_lang=None, target_lang=None,
                 unk_token="<unk>", eos_token="</s>", pad_token="<pad>", model_max_length=512, **kwargs):
        super().__init__(vocab, source_spm, target_spm, source_lang, target_lang,
                         unk_token, eos_token, pad_token, model_max_length, **kwargs)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Uses target language sentencepiece model"""
        return self.current_spm.DecodePieces(tokens)


###############


class GeniePreTrainedModel(PreTrainedModel):
    """
    General class for PreTrainedModel which can output cross-attention weights during generation
    """
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

        # build the tuple to return
        outputs = (input_ids,)

        if output_attentions:
            # List of each encoder layer cross-attention values each with size (bsz, num_heads, tgt_len, src_len)
            all_cross_attentions = [layer_all_cross_attentions[:, :, :sequence_lengths.max().item(), :] for
                                    layer_all_cross_attentions in all_cross_attentions]
            outputs += (all_cross_attentions,)
        
        # TODO change callers to always accept a tuple
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
    
    
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

###############
## NER models

###############
## BERT

class BertEmbeddingsForNER(BertEmbeddings):
    """Construct the embeddings from word, position, token_type, and entity_type embeddings.
    """

    def __init__(self, config, num_db_types, db_unk_id):
        super().__init__(config)
        self.num_db_types = num_db_types
        self.db_unk_id = db_unk_id
        self.pad_token_id = config.pad_token_id
        if num_db_types > 0:
            self.entity_type_embeddings = nn.Embedding(num_db_types, config.hidden_size, padding_idx=db_unk_id)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
                entity_ids=None, entity_masking=None, entity_probs=None, entity_word_embeds_dropout=False):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # dropout word embeddings for entities
        if entity_masking is not None and entity_word_embeds_dropout > 0.0:
            if entity_word_embeds_dropout > random.random():
                input_ids = input_ids * ~(entity_masking.max(-1)[0].bool())

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        if self.num_db_types > 0 and entity_ids is not None:
            # get length for unpadded types
            type_lengths = entity_masking.sum(-1)
            
            # avoid division by zero
            type_lengths[type_lengths == 0] = 1
            
            entity_type_embeddings = self.entity_type_embeddings(entity_ids)
            
            # weighted average
            if entity_probs is not None:
                entity_type_embeddings = entity_type_embeddings * entity_probs.unsqueeze(-1)
            
            # average embedding of different types
            # size (batch, length, num_types, emb_dim)
            entity_type_embeddings = entity_type_embeddings.sum(-2) / type_lengths.unsqueeze(-1)
            
            embeddings += entity_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModelForNER(BertModel):
    """
    Subcalss of BertModel model with an additional entity type embedding layer in the bottom
    """
    def __init__(self, config, num_db_types, db_unk_id):
        super().__init__(config)

        self.embeddings = BertEmbeddingsForNER(config, num_db_types, db_unk_id)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            entity_ids=None,
            entity_masking=None,
            entity_probs=None,
            entity_word_embeds_dropout=False,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_output=None,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        if embedding_output is None:
            embedding_output = self.embeddings(
                input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
                entity_ids=entity_ids, entity_masking=entity_masking, entity_probs=entity_probs, entity_word_embeds_dropout=entity_word_embeds_dropout
            )
            
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    
###############
## Roberta

class RobertaEmbeddingsForNER(BertEmbeddingsForNER):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing and adding entity_types.
    """

    def __init__(self, config, num_db_types, db_unk_id):
        super().__init__(config, num_db_types, db_unk_id)
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
                entity_ids=None, entity_masking=None, entity_probs=None, entity_word_embeds_dropout=False):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds,
            entity_ids=entity_ids, entity_masking=entity_masking, entity_probs=entity_probs, entity_word_embeds_dropout=entity_word_embeds_dropout)

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaModelForNER(RobertaPreTrainedModel):
    def __init__(self, config, num_db_types, db_unk_id,  add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddingsForNER(config, num_db_types, db_unk_id)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            entity_ids=None,
            entity_masking=None,
            entity_probs=None,
            entity_word_embeds_dropout=False,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_output=None,
    ):
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
    
        device = input_ids.device if input_ids is not None else inputs_embeds.device
    
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
    
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
    
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    
        if embedding_output is None:
            embedding_output = self.embeddings(
                input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                entity_ids=entity_ids, entity_masking=entity_masking, entity_probs=entity_probs,
                entity_word_embeds_dropout=entity_word_embeds_dropout
            )
    
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
    
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
    
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class XLMRobertaModelForNER(RobertaModelForNER):
    """
    Subcalss of XLMRobertaModel model with an additional entity type embedding layer at the bottom
    """

    config_class = XLMRobertaConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_db_types, db_unk_id, add_pooling_layer=True):
        super().__init__(config, num_db_types, db_unk_id, add_pooling_layer)
