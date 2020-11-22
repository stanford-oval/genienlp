import re
import torch
import torch.nn.functional as F
from typing import List, Optional


from transformers import LogitsProcessorList
from transformers import MarianMTModel, BartForConditionalGeneration, MBartForConditionalGeneration,\
    T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel

from transformers.models.mbart.tokenization_mbart import MBartTokenizer, _all_mbart_models, SPM_URL

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


MARIAN_SUPPORTED_LANGUAGES = ['https://huggingface.co/Helsinki-NLP']

# all MarianMT models use the same config
MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Helsinki-NLP/opus-mt-en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP/opus-mt-en-de/config.json",
}

MARIAN_GROUP_MEMBERS = {
    'ZH': ['cmn', 'cn', 'yue', 'ze_zh', 'zh_cn', 'zh_CN', 'zh_HK', 'zh_tw', 'zh_TW', 'zh_yue', 'zhs', 'zht', 'zh'],
    'ROMANCE': ['fr', 'fr_BE', 'fr_CA', 'fr_FR', 'wa', 'frp', 'oc', 'ca', 'rm', 'lld', 'fur', 'lij', 'lmo',
                'es', 'es_AR', 'es_CL', 'es_CO', 'es_CR', 'es_DO', 'es_EC', 'es_ES', 'es_GT', 'es_HN', 'es_MX', 'es_NI', 'es_PA', 'es_PE', 'es_PR', 'es_SV', 'es_UY', 'es_VE',
                'pt', 'pt_br', 'pt_BR', 'pt_PT', 'gl', 'lad', 'an', 'mwl', 'it', 'it_IT', 'co', 'nap', 'scn', 'vec', 'sc', 'ro', 'la'],
    'NORTH_EU': ['de', 'nl', 'fy', 'af', 'da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
    'SCANDINAVIA': ['da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
    'SAMI': ['se', 'sma', 'smj', 'smn', 'sms'],
    'NORWAY': ['nb_NO', 'nb', 'nn_NO', 'nn', 'nog', 'no_nb', 'no'],
    'CELTIC': ['ga', 'cy', 'br', 'gd', 'kw', 'gv'],
    'trk': ["aze_Latn", "bak", "chv", "crh", "crh_Latn", "kaz_Cyrl", "kaz_Latn", "kir_Cyrl", "kjh", "kum", "ota_Arab",
            "ota_Latn", "sah", "tat", "tat_Arab", "tat_Latn", "tuk", "tuk_Latn", "tur", "tyv", "uig_Arab", "uig_Cyrl", "uzb_Cyrl", "uzb_Latn"],
    'mul': ["abk", "acm", "ady", "afb", "afh_Latn", "afr", "akl_Latn", "aln", "amh", "ang_Latn", "apc", "ara", "arg", "arq", "ary",
            "arz", "asm", "ast", "avk_Latn", "awa", "aze_Latn", "bak", "bam_Latn", "bel", "bel_Latn", "ben", "bho", "bod",
            "bos_Latn", "bre", "brx", "brx_Latn", "bul", "bul_Latn", "cat", "ceb", "ces", "cha", "che", "chr", "chv",
            "cjy_Hans", "cjy_Hant", "cmn", "cmn_Hans", "cmn_Hant", "cor", "cos", "crh", "crh_Latn", "csb_Latn", "cym",
            "dan", "deu", "dsb", "dtp", "dws_Latn", "egl", "ell", "enm_Latn", "epo", "est", "eus", "ewe", "ext", "fao",
            "fij", "fin", "fkv_Latn", "fra", "frm_Latn", "frr", "fry", "fuc", "fuv", "gan", "gcf_Latn", "gil", "gla",
            "gle", "glg", "glv", "gom", "gos", "got_Goth", "grc_Grek", "grn", "gsw", "guj", "hat", "hau_Latn", "haw",
            "heb", "hif_Latn", "hil", "hin", "hnj_Latn", "hoc", "hoc_Latn", "hrv", "hsb", "hun", "hye", "iba", "ibo",
            "ido", "ido_Latn", "ike_Latn", "ile_Latn", "ilo", "ina_Latn", "ind", "isl", "ita", "izh", "jav", "jav_Java",
            "jbo", "jbo_Cyrl", "jbo_Latn", "jdt_Cyrl", "jpn", "kab", "kal", "kan", "kat", "kaz_Cyrl", "kaz_Latn", "kek_Latn",
            "kha", "khm", "khm_Latn", "kin", "kir_Cyrl", "kjh", "kpv", "krl", "ksh", "kum", "kur_Arab", "kur_Latn", "lad",
            "lad_Latn", "lao", "lat_Latn", "lav", "ldn_Latn", "lfn_Cyrl", "lfn_Latn", "lij", "lin", "lit", "liv_Latn", "lkt",
            "lld_Latn", "lmo", "ltg", "ltz", "lug", "lzh", "lzh_Hans", "mad", "mah", "mai", "mal", "mar", "max_Latn", "mdf",
            "mfe", "mhr", "mic", "min", "mkd", "mlg", "mlt", "mnw", "moh", "mon", "mri", "mwl", "mww", "mya", "myv", "nan",
            "nau", "nav", "nds", "niu", "nld", "nno", "nob", "nob_Hebr", "nog", "non_Latn", "nov_Latn", "npi", "nya", "oci",
            "ori", "orv_Cyrl", "oss", "ota_Arab", "ota_Latn", "pag", "pan_Guru", "pap", "pau", "pdc", "pes", "pes_Latn",
            "pes_Thaa", "pms", "pnb", "pol", "por", "ppl_Latn", "prg_Latn", "pus", "quc", "qya", "qya_Latn", "rap",
            "rif_Latn", "roh", "rom", "ron", "rue", "run", "rus", "sag", "sah", "san_Deva", "scn", "sco", "sgs",
            "shs_Latn", "shy_Latn", "sin", "sjn_Latn", "slv", "sma", "sme", "smo", "sna", "snd_Arab", "som", "spa", "sqi",
            "srp_Cyrl", "srp_Latn", "stq", "sun", "swe", "swg", "swh", "tah", "tam", "tat", "tat_Arab", "tat_Latn", "tel",
            "tet", "tgk_Cyrl", "tha", "tir", "tlh_Latn", "tly_Latn", "tmw_Latn", "toi_Latn", "ton", "tpw_Latn", "tso", "tuk",
            "tuk_Latn", "tur", "tvl", "tyv", "tzl", "tzl_Latn", "udm", "uig_Arab", "uig_Cyrl", "ukr", "umb", "urd", "uzb_Cyrl",
            "uzb_Latn", "vec", "vie", "vie_Hani", "vol_Latn", "vro", "war", "wln", "wol", "wuu", "xal", "xho", "yid", "yor",
            "yue", "yue_Hans", "yue_Hant", "zho", "zho_Hans", "zho_Hant", "zlm_Latn", "zsm_Latn", "zul", "zza"]
}


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
