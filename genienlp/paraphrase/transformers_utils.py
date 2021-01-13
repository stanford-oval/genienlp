import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

import numpy as np

from transformers import XLMRobertaConfig, BartConfig

from transformers import LogitsProcessorList
from transformers import MarianMTModel, BartForConditionalGeneration, MBartForConditionalGeneration,\
    T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutput, \
    Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bart.modeling_bart import BartEncoder, BartModel, shift_tokens_right, _expand_mask

from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertPreTrainedModel, BertEmbeddings, BertModel
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids, RobertaEncoder, \
    RobertaPooler

from transformers.models.marian.convert_marian_to_pytorch import GROUPS
from transformers.models.marian.convert_marian_tatoeba_to_pytorch import GROUP_MEMBERS

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

###############
## NER models


###############
## BART

class BartForConditionalGenerationForNER(BartForConditionalGeneration):

    def __init__(self, config: BartConfig, num_db_types, db_unk_id):
        super().__init__(config)
        self.model = BartModelForNER(config, num_db_types, db_unk_id)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        entity_ids=None,
        entity_masking=None,
        entity_probs=None,
        mask_entities=False,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            entity_ids=entity_ids,
            entity_masking=entity_masking,
            entity_probs=entity_probs,
            mask_entities=mask_entities,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class BartModelForNER(BartModel):
    def __init__(self, config: BartConfig, num_db_types, db_unk_id):
        super().__init__(config)

        self.encoder = BartEncoderForNER(config, num_db_types, db_unk_id, self.shared)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        entity_ids=None,
        entity_masking=None,
        entity_probs=None,
        mask_entities=False,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # 4.12.20 (PVP): Not a fan of this "magical" function and
        # also wonder how often it's actually used ... keep now
        # for backward compatibility
        # -> is this used for backward compatibility
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                entity_ids=entity_ids,
                entity_masking=entity_masking,
                entity_probs=entity_probs,
                mask_entities=mask_entities,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )




class BartEncoderForNER(BartEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, num_db_types, db_unk_id, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.num_db_types = num_db_types
        self.db_unk_id = db_unk_id
        self.pad_token_id = config.pad_token_id
        if num_db_types > 0:
            self.entity_type_embeddings = nn.Embedding(num_db_types, config.hidden_size, padding_idx=db_unk_id)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        entity_ids=None,
        entity_masking=None,
        entity_probs=None,
        mask_entities=False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.
                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos

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
    
            hidden_states += entity_type_embeddings
        
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                hidden_states, attn = encoder_layer(hidden_states, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

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

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
                entity_ids=None, entity_masking=None, entity_probs=None, mask_entities=False):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
            
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if entity_masking is not None and mask_entities:
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
            mask_entities=False,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
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
        
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
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
                entity_ids=entity_ids, entity_masking=entity_masking, entity_probs=entity_probs, mask_entities=mask_entities
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
                entity_ids=None, entity_masking=None, entity_probs=None, mask_entities=False):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length,
            entity_ids=entity_ids, entity_masking=entity_masking, entity_probs=entity_probs, mask_entities=mask_entities)

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

class XLMRobertaModelForNER(BertModelForNER):
    """
    Subcalss of XLMRobertaModel model with an additional entity type embedding layer at the bottom
    """

    config_class = XLMRobertaConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_db_types, db_unk_id, add_pooling_layer=True):
        super().__init__(config, num_db_types, db_unk_id)

        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        
        self.embeddings = RobertaEmbeddingsForNER(config, num_db_types, db_unk_id)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        
###############

class BootlegBertEncoder(BertPreTrainedModel):
    def __init__(
        self,
        config,
        input_dim,
        output_dim,
        ent_emb_file=None,
        static_ent_emb_file=None,
        type_ent_emb_file=None,
        rel_ent_emb_file=None,
        tanh=False,
        norm=False,
        freeze=True,
    ):
        super(BootlegBertEncoder, self).__init__(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.apply(self._init_weights)

        if ent_emb_file is not None:
            ent_emb_matrix = torch.from_numpy(np.load(ent_emb_file))
            self.ent_embeddings = nn.Embedding(
                ent_emb_matrix.size()[0], ent_emb_matrix.size()[1], padding_idx=0
            )
            self.ent_embeddings.weight.data.copy_(ent_emb_matrix)
            input_dim += ent_emb_matrix.size()[1]
            if freeze:
                for param in self.ent_embeddings.parameters():
                    param.requires_grad = False
        else:
            self.ent_embeddings = None

        if static_ent_emb_file is not None:
            static_ent_emb_matrix = torch.from_numpy(np.load(static_ent_emb_file))
            self.static_ent_embeddings = nn.Embedding(
                static_ent_emb_matrix.size()[0],
                static_ent_emb_matrix.size()[1],
                padding_idx=0,
            )
            self.static_ent_embeddings.weight.data.copy_(static_ent_emb_matrix)
            input_dim += static_ent_emb_matrix.size()[1]
            if freeze:
                for param in self.static_ent_embeddings.parameters():
                    param.requires_grad = False
        else:
            self.static_ent_embeddings = None

        if type_ent_emb_file is not None:
            type_ent_emb_matrix = torch.from_numpy(np.load(type_ent_emb_file))
            self.type_ent_embeddings = nn.Embedding(
                type_ent_emb_matrix.size()[0],
                type_ent_emb_matrix.size()[1],
                padding_idx=0,
            )
            self.type_ent_embeddings.weight.data.copy_(type_ent_emb_matrix)
            input_dim += type_ent_emb_matrix.size()[1]
            if freeze:
                for param in self.type_ent_embeddings.parameters():
                    param.requires_grad = False
        else:
            self.type_ent_embeddings = None

        if rel_ent_emb_file is not None:
            rel_ent_emb_matrix = torch.from_numpy(np.load(rel_ent_emb_file))
            self.rel_ent_embeddings = nn.Embedding(
                rel_ent_emb_matrix.size()[0],
                rel_ent_emb_matrix.size()[1],
                padding_idx=0,
            )
            self.rel_ent_embeddings.weight.data.copy_(rel_ent_emb_matrix)
            input_dim += rel_ent_emb_matrix.size()[1]
            if freeze:
                for param in self.rel_ent_embeddings.parameters():
                    param.requires_grad = False
        else:
            self.rel_ent_embeddings = None

        self.proj = nn.Linear(input_dim, output_dim)

        if tanh is True:
            self.proj_activation = nn.Tanh()
        else:
            self.proj_activation = None

        self.norm = norm
        if self.norm is True:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        inputs_embeds,
        input_ids=None,
        input_ent_ids=None,
        input_static_ent_ids=None,
        input_type_ent_ids=None,
        input_rel_ent_ids=None,
        token_type_ids=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=True,
    ):
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
    
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
    
        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
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
        
        embedding_output = [inputs_embeds]

        ents_embeddings = None
        if self.ent_embeddings is not None:
            # input_ent_ids of size (batch, length, num_ids)
            ents_embeddings = self.ent_embeddings(input_ent_ids)
            
            # average them
            ents_embeddings = ents_embeddings.mean(-2)
            
            embedding_output.append(ents_embeddings)

        static_ents_embeddings = None
        if self.static_ent_embeddings is not None:
            static_ents_embeddings = self.static_ent_embeddings(input_static_ent_ids)
            embedding_output.append(static_ents_embeddings)

        type_ents_embeddings = None
        if self.type_ent_embeddings is not None:
            type_ents_embeddings = self.type_ent_embeddings(input_type_ent_ids)
            embedding_output.append(type_ents_embeddings)

        rel_ents_embeddings = None
        if self.rel_ent_embeddings is not None:
            rel_ents_embeddings = self.rel_ent_embeddings(input_rel_ent_ids)
            embedding_output.append(rel_ents_embeddings)

        embedding_output = torch.cat(embedding_output, dim=-1)
        embedding_output = self.proj(embedding_output)

        if self.proj_activation:
            embedding_output = self.proj_activation(embedding_output)

        if self.norm:
            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
    
    
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
    
        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:
                                                      ]  # add hidden_states and attentions if they are here

        
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

###############


# from ..models.common import Feedforward
#
# class BootlegEmbedding(torch.nn.Module):
#     def __init__(self, ent_emb_file, output_dim, freeze=True):
#         super().__init__()
#         self.ent_emb_file = ent_emb_file
#         ent_emb_matrix = torch.from_numpy(np.load(ent_emb_file))
#         self.ent_embeddings = torch.nn.Embedding(*ent_emb_matrix.size(), padding_idx=0)
#         self.ent_embeddings.weight.data.copy_(ent_emb_matrix)
#
#         if freeze:
#             for param in self.ent_embeddings.parameters():
#                 param.requires_grad = False
#         self.ent_proj = Feedforward(ent_emb_matrix.size()[1], output_dim, bias=False)
#
#     def forward(self, input_ent_ids: torch.Tensor):
#         ents_embeddings = self.ent_embeddings(input_ent_ids)
#         ents_embeddings = self.ent_proj(ents_embeddings)
#
#         return ents_embeddings
