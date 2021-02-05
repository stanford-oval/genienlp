# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from collections import defaultdict
import marisa_trie
import ujson

from ..data_utils.database_utils import DOMAIN_TYPE_MAPPING
from ..data_utils.remote_database import RemoteElasticDatabase
from ..tasks.base_dataset import Split
from ..tasks.base_task import BaseTask
from ..tasks.generic_dataset import input_then_output_len, input_tokens_fn, CQA, default_batch_fn
from ..tasks.registry import register_task
from ..data_utils.database import Database
from ..data_utils.example import Example, get_pad_feature, Feature
from ..tasks.almond_dataset import AlmondDataset
from ..tasks.almond_utils import ISO_to_LANG, process_id, quoted_pattern_with_space, \
    tokenize_cjk_chars, detokenize_cjk_chars, is_entity, is_entity_marker, is_device
from ..paraphrase.data_utils import input_heuristics, output_heuristics



class BaseAlmondTask(BaseTask):
    """Base class for the Almond semantic parsing task
        i.e. natural language to formal language (ThingTalk) mapping"""
    
    def __init__(self, name, args):
        super().__init__(name, args)
        self.args = args
        no_feature_fields = ['answer']
        if self.is_contextual():
            no_feature_fields.append('context')
        else:
            no_feature_fields.append('question')
        self.no_feature_fields = no_feature_fields
        
        self._almond_has_multiple_programs = args.almond_has_multiple_programs
        self._almond_detokenize_sentence = args.almond_detokenize_sentence
        self._almond_thingtalk_version = args.almond_thingtalk_version
        
        self.almond_domains = args.almond_domains
        self.all_schema_types = set()

        if args.do_ner:
            self.unk_id = args.features_default_val[0]
            self.TTtype2DBtype = dict()
    
            for domain in self.almond_domains:
                self.TTtype2DBtype.update(DOMAIN_TYPE_MAPPING[domain])
            self.DBtype2TTtype = {v: k for k, v in self.TTtype2DBtype.items()}
            self._init_db()

    def _init_db(self):
        if self.args.database_type in ['json']:
            with open(os.path.join(self.args.database_dir, 'es_material/canonical2type.json'), 'r') as fin:
                canonical2type = ujson.load(fin)
            with open(os.path.join(self.args.database_dir, 'es_material/type2id.json'), 'r') as fin:
                type2id = ujson.load(fin)
            all_canonicals = marisa_trie.Trie(canonical2type.keys())

            self.db = Database(canonical2type, type2id, all_canonicals,
                               self.args.features_default_val, self.args.features_size)
        
        elif self.args.database_type == 'remote-elastic':
            with open(os.path.join(self.args.database_dir, 'es_material/elastic_config.json'), 'r') as fin:
                es_config = ujson.load(fin)
            with open(os.path.join(self.args.database_dir, 'es_material/type2id.json'), 'r') as fin:
                type2id = ujson.load(fin)
            self.db = RemoteElasticDatabase(es_config, type2id, self.args.features_default_val, self.args.features_size)

    @property
    def is_contextual(self):
        return NotImplementedError
    
    @property
    def metrics(self):
        return ['em', 'sm', 'f1']
    
    def _is_program_field(self, field_name):
        raise NotImplementedError()
    
    def _make_example(self, parts, dir_name, **kwargs):
        raise NotImplementedError()
    
    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, **kwargs)
    
    
    def collect_answer_entity_types(self, answer):
        entity2type = dict()
        
        answer_entities = quoted_pattern_with_space.findall(answer)
        for ent in answer_entities:
            
            
            if self._almond_thingtalk_version == 1:
                #  ... param:inAlbum:Entity(org.schema.Music:MusicAlbum) == " XXXX " ...
                # ... param:artists contains " XXX " ^^com.spotify:artist and param:id =~ " XXX " ...
                # ... filter param:geo:Location == location: " XXXX " and ...
    
                # assume first syntax
                idx = answer.index('" ' + ent + ' "')
    
                schema_entity_type = None
    
                answer_tokens_after_entity = answer[idx + len('" ' + ent + ' "'):].split()
                if answer_tokens_after_entity[0].startswith('^^'):
                    schema_entity_type = answer_tokens_after_entity[0].rsplit(':', 1)[1]
    
                if schema_entity_type is None:
        
                    answer_tokens_before_entity = answer[:idx].split()
        
                    # check last three tokens to find one that starts with param
                    for i in range(3):
                        if answer_tokens_before_entity[-i].startswith('param:'):
                            schema_entity_type = answer_tokens_before_entity[-i]
                            break
        
                    if schema_entity_type:
                        schema_entity_type = schema_entity_type.strip('()').rsplit(':', 1)[1]
    
                if schema_entity_type is None or schema_entity_type not in self.TTtype2DBtype.keys():
                    schema_type = self.db.unk_type
                else:
                    schema_type = self.TTtype2DBtype[schema_entity_type]
    
                entity2type[ent] = schema_type

            else:
            
                # ** this should change if thingtalk syntax changes **
                
                # ( ... [Book|Music|...] ( ) filter id =~ " position and affirm " ) ...'
                # ... ^^org.schema.Book:Person ( " james clavell " ) ...
                # ... contains~ ( [award|genre|...] , " booker " ) ...
                # ... inLanguage =~ " persian " ...
                
                
                # missing syntax from current code
                #  ... @com.spotify . song ( ) filter in_array~ ( id , [ " piano man " , " uptown girl " ] ) ) [ 1 : 2 ]  ...
                
                # assume first syntax
                idx = answer.index('" ' + ent + ' "')
                
                schema_entity_type = None
                
                tokens_after_entity = answer[idx + len('" ' + ent + ' "'):].split()
                tokens_before_entity = answer[:idx].split()
                
                if tokens_before_entity[-2] == 'Location':
                    schema_entity_type = 'Location'
                
                elif tokens_before_entity[-1] in ['>=', '==', '<='] and tokens_before_entity[-2] in ['ratingValue',
                                                                                                     'reviewCount',
                                                                                                     'checkoutTime',
                                                                                                     'checkinTime']:
                    schema_entity_type = tokens_before_entity[-2]
                
                elif tokens_before_entity[-1] == '=~':
                    if tokens_before_entity[-2] == 'id':
                        # travers previous tokens until find filter
                        i = -3
                        while tokens_before_entity[i] != 'filter' and i > 3-len(tokens_before_entity):
                            i -= 1
                        schema_entity_type = tokens_before_entity[i-3]
                    else:
                        schema_entity_type = tokens_before_entity[-2]
                
                elif tokens_before_entity[-4] == 'contains~':
                    schema_entity_type = tokens_before_entity[-2]
                
                elif tokens_before_entity[-2].startswith('^^'):
                    schema_entity_type = tokens_before_entity[-2].rsplit(':', 1)[1]
                    if schema_entity_type == 'Person' and tokens_before_entity[-3] == 'null' and tokens_before_entity[-5] in ['director', 'creator', 'actor']:
                        schema_entity_type = 'Person.' + tokens_before_entity[-5]
                
                if schema_entity_type is None or schema_entity_type not in self.TTtype2DBtype.keys():
                    schema_type = self.db.unk_type
                else:
                    schema_type = self.TTtype2DBtype[schema_entity_type]
                
                entity2type[ent] = schema_type
                
            self.all_schema_types.add(schema_entity_type)
            
        return entity2type
    
    def pad_features(self, features, max_size, pad_id):
        if len(features) > max_size:
            features = features[:max_size]
        else:
            features += [pad_id] * (max_size - len(features))
        return features
    
    def oracle_type_ids(self, tokens, entity2type):
        tokens_type_ids = [[self.args.features_default_val[0]] * self.args.features_size[0] for _ in range(len(tokens))]
        tokens_text = " ".join(tokens)
        
        for ent, type in entity2type.items():
            ent_num_tokens = len(ent.split(' '))
            idx = tokens_text.index(ent)
            token_pos = len(tokens_text[:idx].split())
            
            type_id = self.db.type2id[type]
            type_id = self.pad_features([type_id], self.args.features_size[0], self.args.features_default_val[0])
            
            tokens_type_ids[token_pos: token_pos + ent_num_tokens] = [type_id] * ent_num_tokens
        
        return tokens_type_ids
    
    def find_type_ids(self, tokens, answer):
        tokens_type_ids = []
        
        if self.args.database_type == 'json':
            if self.args.retrieve_method == 'naive':
                tokens_type_ids = self.db.lookup(tokens, self.args.lookup_method, self.args.min_entity_len,
                                                 self.args.max_entity_len)
            elif self.args.retrieve_method == 'entity-oracle':
                answer_entities = quoted_pattern_with_space.findall(answer)
                tokens_type_ids = self.db.lookup(tokens, answer_entities=answer_entities)
            elif self.args.retrieve_method == 'type-oracle':
                entity2type = self.collect_answer_entity_types(answer)
                tokens_type_ids = self.oracle_type_ids(tokens, entity2type)
        
        return tokens_type_ids


    def find_type_probs(self, tokens, default_val, default_size):
        token_freqs = [[default_val] * default_size] * len(tokens)
        return token_freqs

    def postprocess_prediction(self, example_id, prediction):
        
        if self._almond_detokenize_sentence:
            # To make genienlp transparent to the tokenization done by genie-toolkit
            # We tokenize prediction here by adding whitespace between each CJK character
            prediction = tokenize_cjk_chars(prediction)
            
        new_tokens = []
        for token in prediction.split():
            if token.startswith('STRING_'):
                token = 'QUOTED_' + token
            elif token.startswith('ENTITY_'):
                token = 'GENERIC_' + token
            new_tokens.append(token)
            
        new_prediction = ' '.join(new_tokens)
        return new_prediction
    
    def create_sentence_plus_types_tokens(self, new_sentence, features, add_types_to_text):
        new_sentence_tokens = new_sentence.split(' ')
        assert len(new_sentence_tokens) == len(features)
        sentence_plus_types_tokens = []
        i = 0
        if add_types_to_text == 'insert':
            while i < len(new_sentence_tokens):
                token = new_sentence_tokens[i]
                feat = features[i]
                # token is an entity
                if any([val != self.args.features_default_val[0] for val in feat.type_id]):
                    final_token = '<e> '
                    all_types = ' | '.join(set([self.DBtype2TTtype[self.db.id2type[id]] for id in feat.type_id if self.db.id2type[id] in self.DBtype2TTtype]))
                    final_token += '( ' + all_types + ' ) ' + token
                    # append all entities with same type
                    i += 1
                    while i < len(new_sentence_tokens) and features[i] == feat:
                        final_token += ' ' + new_sentence_tokens[i]
                        i += 1
                    final_token += ' </e>'
                    sentence_plus_types_tokens.append(final_token)
                else:
                    sentence_plus_types_tokens.append(token)
                    i += 1
        
        
        elif add_types_to_text == 'append':
            sentence_plus_types_tokens.extend(new_sentence_tokens)
            sentence_plus_types_tokens.append('<e>')
            while i < len(new_sentence_tokens):
                feat = features[i]
                # token is an entity
                if any([val != self.args.features_default_val[0] for val in feat.type_id]):
                    all_types = ' | '.join(set([self.DBtype2TTtype[self.db.id2type[id]] for id in feat.type_id if self.db.id2type[id] in self.DBtype2TTtype]))
                    all_tokens = []
                    # append all entities with same type
                    while i < len(new_sentence_tokens) and features[i] == feat:
                        all_tokens.append(new_sentence_tokens[i])
                        i += 1
                    final_token = ' '.join([*all_tokens, '(', all_types, ')', ';'])
                    sentence_plus_types_tokens.append(final_token)
                else:
                    i += 1

            sentence_plus_types_tokens.append('</e>')

        return ' '.join(sentence_plus_types_tokens)

    def preprocess_field(self, sentence, field_name=None, answer=None):
        
        if self.override_context is not None and field_name == 'context':
            pad_feature = get_pad_feature(self.args.features, self.args.features_default_val, self.args.features_size)
            return self.override_context, [pad_feature] * len(self.override_context.split(' ')), self.override_context
        if self.override_question is not None and field_name == 'question':
            pad_feature = get_pad_feature(self.args.features, self.args.features_default_val, self.args.features_size)
            return self.override_question, [pad_feature] * len(self.override_question.split(' ')), self.override_question
        if not sentence:
            return '', [], ''
        
        tokens = sentence.split(' ')
        is_program = self._is_program_field(field_name)
        new_tokens = []
        for token in tokens:
            if is_entity(token) or (is_program and (is_device(token) or is_entity_marker(token))):
                if token.startswith('QUOTED_STRING_'):
                    token = token[len('QUOTED_'):]
                elif token.startswith('GENERIC_ENTITY_'):
                    token = token[len('GENERIC_'):]
                
                self.special_tokens.add(token)
            new_tokens.append(token)
        tokens = new_tokens
        new_sentence = ' '.join(tokens)
        
        if self._almond_detokenize_sentence:
            
            # BERT tokenizers by default add whitespace around any CJK character
            # SPM-based tokenizers are trained on raw text and do better when recieve untokenized text
            # In genienlp we detokenize CJK characters and leave tokenization to the model's tokenizer
            # NOTE: input datasets for almond are usually pretokenized using genie-toolkit which
            # inserts whitespace around any CJK character. This detokenization ensures that SPM-based tokenizers
            # see the text without space between those characters
            new_sentence = detokenize_cjk_chars(new_sentence)
            tokens = new_sentence.split(' ')
            
            new_sentence = ''
            in_string = False
            for token in tokens:
                if is_program:
                    if token == '"':
                        in_string = not in_string
                    if not in_string:
                        new_sentence += ' ' + token
                        continue
                if token in (',', '.', '?', '!', ':', ')', ']', '}') or token.startswith("'"):
                    new_sentence += token
                else:
                    new_sentence += ' ' + token
        elif is_program and field_name != 'answer':
            new_tokens = []
            in_string = False
            for token in tokens:
                if token == '"':
                    in_string = not in_string
                if in_string:
                    new_tokens.append(token)
                    continue
                
                if not is_entity(token) and not is_entity_marker(token) and \
                        not is_device(token):
                    for word in token.split('_'):
                        new_tokens.append(word)
                else:
                    new_tokens.append(token)
            new_sentence = ' '.join(new_tokens)
        
        new_sentence = new_sentence.strip()
        new_tokens = new_sentence.split(' ')
        new_sentence_length = len(new_tokens)
        
        tokens_type_ids, tokens_type_probs = None, None
        
        if 'type_id' in self.args.features and field_name != 'answer':
            tokens_type_ids = [[self.args.features_default_val[0]] * self.args.features_size[0] for _ in
                               range(new_sentence_length)]
        if 'type_prob' in self.args.features and field_name != 'answer':
            tokens_type_probs = [[self.args.features_default_val[1]] * self.args.features_size[1] for _ in
                                 range(new_sentence_length)]
        
        if self.args.do_ner and self.args.retrieve_method != 'bootleg' and field_name not in self.no_feature_fields:
            if 'type_id' in self.args.features:
                tokens_type_ids = self.find_type_ids(new_tokens, answer)
            if 'type_prob' in self.args.features:
                tokens_type_probs = self.find_type_probs(new_tokens, self.args.features_default_val[1],
                                                         self.args.features_size[1])
            
            if self.args.verbose and self.args.do_ner:
                print()
                print(
                    *[f'token: {token}\ttype: {token_type}' for token, token_type in zip(new_tokens, tokens_type_ids)],
                    sep='\n')
            else:
                # TODO move bootleg out of almond_dataset to here
                pass
        
        zip_list = []
        if tokens_type_ids:
            assert len(tokens_type_ids) == new_sentence_length
            zip_list.append(tokens_type_ids)
        if tokens_type_probs:
            assert len(tokens_type_probs) == new_sentence_length
            zip_list.append(tokens_type_probs)
        
        features = [Feature(*tup) for tup in zip(*zip_list)]

        sentence_plus_types = ''
        if self.args.do_ner and self.args.add_types_to_text != 'no' and len(features):
            sentence_plus_types = self.create_sentence_plus_types_tokens(new_sentence, features, self.args.add_types_to_text)

        return new_sentence, features, sentence_plus_types


@register_task('almond')
class Almond(BaseAlmondTask):
    """The Almond semantic parsing task
    i.e. natural language to formal language (ThingTalk) mapping"""
    
    def _is_program_field(self, field_name):
        return field_name == 'answer'
    
    def is_contextual(self):
        return False
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        if self._almond_has_multiple_programs:
            example_id, sentence, target_code = parts[:3]
        else:
            example_id, sentence, target_code = parts
        question = 'translate from english to thingtalk'
        context = sentence
        answer = target_code
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)


@register_task('natural_seq2seq')
class NaturalSeq2Seq(BaseAlmondTask):
    """The Almond seqeunce to sequence task where both sequences are natural language
    i.e. no ThingTalk program. Paraphrasing and translation are examples of this task"""
    
    @property
    def metrics(self):
        return ['bleu', 'em', 'nf1']
    
    def _is_program_field(self, field_name):
        return False
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant
        if len(parts) == 2:
            input_sequence, target_sequence = parts
            example_id = "id-null"
        elif len(parts) == 3:
            example_id, input_sequence, target_sequence = parts
        else:
            raise ValueError(f'Input file contains line with {len(parts)} parts: {str(parts)}')
        question = 'translate from input to output'
        context = input_sequence
        answer = target_sequence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, **kwargs)


@register_task('paraphrase')
class Paraphrase(NaturalSeq2Seq):
    """The Almond paraphrasing task. Applies the necessary preprocessing for special tokens and case changes.
    Should only be used at prediction time.
    """

    def __init__(self, name, args):
        super().__init__(name, args)
        self.reverse_maps = {}

    @property
    def metrics(self):
        return ['bleu']

    def postprocess_prediction(self, example_id, prediction):
        return output_heuristics(prediction, self.reverse_maps[example_id])


    def _make_example(self, parts, dir_name=None, **kwargs):
        if len(parts) == 3:
            example_id, sentence, thingtalk = parts
        elif len(parts) == 4:
            example_id, _, sentence, thingtalk = parts # ignore dialogue context
        else:
            raise ValueError(f'Input file contains line with {len(parts)} parts: {str(parts)}')

        example_id = self.name + '/' + example_id

        sentence, reverse_map = input_heuristics(sentence, thingtalk=thingtalk, is_cased=True)
        # this task especially needs example ids to be unique
        if example_id in self.reverse_maps:
            example_id += '.'
        self.reverse_maps[example_id] = reverse_map

        question = 'translate from input to output'
        context = sentence
        answer = sentence # means we calculate self-bleu
        
        return Example.from_raw(example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)
    
    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond'), make_example=self._make_example, **kwargs)


@register_task('contextual_almond')
class ContextualAlmond(BaseAlmondTask):
    """Contextual Almond semantic parsing task
    """
    
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')
    
    def is_contextual(self):
        return True
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts[:4]
        else:
            example_id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)


@register_task('reverse_almond')
class ReverseAlmond(BaseAlmondTask):
    """Reverse Almond semantic parsing task
    i.e. formal language to natural language mapping"""
    
    @property
    def metrics(self):
        return ['bleu', 'em']
    
    def is_contextual(self):
        return False
    
    def _is_program_field(self, field_name):
        return field_name == 'context'
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant, so the question says English and ThingTalk even if we're doing
        # a different language (like Chinese)
        example_id, sentence, target_code = parts
        question = 'translate from thingtalk to english'
        context = target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)


# TODO add a similar preprocessing step to Multilingual dialogue tasks as well
class BaseAlmondDialogueNLUTask(BaseAlmondTask):
    def preprocess_field(self, sentence, field_name=None, answer=None):
        if not sentence:
            return sentence, [], sentence
        
        # remove the $dialogue at the start of the dialogue
        # this is safe because we know we're processing dialogues, so the answer
        # always starts with $dialogue and the context is either `null` or also
        # starts with $dialogue
        if field_name in ['context', 'answer'] and sentence.startswith('$dialogue '):
            sentence = sentence[len('$dialogue '):]
        return super().preprocess_field(sentence, field_name, answer)
    
    def postprocess_prediction(self, example_id, prediction):
        if not prediction.startswith('$'):
            return '$dialogue ' + prediction
        return prediction

@register_task('almond_dialogue_nlu')
class AlmondDialogueNLU(BaseAlmondDialogueNLUTask):
    """Multi-turn NLU task for Almond dialogues
    (translate the user utterance to a formal representation, given the current
    state of the conversation)
    """
    
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')
    
    def is_contextual(self):
        return True
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts[:4]
        else:
            example_id, context, sentence, target_code = parts

        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)
    
    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/user'), make_example=self._make_example,
                                           **kwargs)


@register_task('almond_dialogue_nlu_agent')
class AlmondDialogueNLUAgent(BaseAlmondDialogueNLUTask):
    """Multi-turn NLU task for Almond dialogues, for the agent utterance
    (translate the agent utterance to a formal representation, given the current
    state of the conversation).
    This is used to facilitate annotation of human-human dialogues.
    """
    
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')
    
    def is_contextual(self):
        return True
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts[:4]
        else:
            example_id, context, sentence, target_code = parts
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)
    
    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example,
                                           **kwargs)


@register_task('almond_dialogue_nlg')
class AlmondDialogueNLG(BaseAlmondTask):
    """Multi-turn NLG task for Almond dialogues
    (generate the system utterance, given the current state of the conversation
    and the desired system dialogue act)
    """
    
    def _is_program_field(self, field_name):
        return field_name == 'context'
    
    def is_contextual(self):
        return True
    
    @property
    def metrics(self):
        return ['bleu']
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task
        example_id, context, sentence, target_code = parts
        question = 'what should the agent say ?'
        context = context + ' ' + target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)
    
    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example,
                                           **kwargs)


@register_task('almond_dialogue_policy')
class AlmondDialoguePolicy(BaseAlmondTask):
    """Multi-turn dialogue policy task for Almond dialogues
    (generate the next dialogue act, given the current state of the conversation)
    """
    
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')
    
    def is_contextual(self):
        return True
    
    @property
    def metrics(self):
        return ['em', 'f1']
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task, and the sentence is intentionally ignored
        example_id, context, _sentence, target_code = parts
        question = 'what should the agent do ?'
        context = context
        answer = target_code
        return Example.from_raw(self.name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)
    
    def get_splits(self, root, **kwargs):
        return AlmondDataset.return_splits(path=os.path.join(root, 'almond/agent'), make_example=self._make_example,
                                           **kwargs)


class BaseAlmondMultiLingualTask(BaseAlmondTask):
    """ Base task for MultiLingual Almond
    """
    
    def get_train_processed_ids(self, split):
        all_ids = []
        for ex in split.examples:
            all_ids.append(process_id(ex))
        return all_ids
    
    def combine_datasets(self, datasets, all_paths, sort_key_fn, batch_size_fn, used_fields, groups):
        splits = defaultdict()
        # paths = defaultdict()
        
        for field in used_fields:
            # choose one path and replace dir name with 'combined'
            # paths[field] = '/'.join([getattr(all_paths[0], field).rsplit('/', 2)[0], 'combined', getattr(all_paths[0], field).rsplit('/', 2)[2]])
            
            all_examples = []
            for dataset in datasets:
                all_examples.extend(getattr(dataset, field).examples)

            splits[field] = CQA(all_examples, sort_key_fn=sort_key_fn, batch_size_fn=batch_size_fn, groups=groups)
            
        
        return Split(train=splits.get('train'), eval=splits.get('eval'), test=splits.get('test'), aux=splits.get('aux'))
               # Split(train=paths.get('train'), eval=paths.get('eval'), test=paths.get('test'), aux=paths.get('aux'))


    def get_splits(self, root, **kwargs):
        all_datasets = []
        all_paths = []
        # number of directories to read data from
        all_dirs = kwargs['all_dirs'].split('+')
        
        for dir in all_dirs:
            splits, paths = AlmondDataset.return_splits(path=os.path.join(root, 'almond/multilingual/{}'.format(dir)),
                                                         make_example=self._make_example, **kwargs)
            all_datasets.append(splits)
            all_paths.append(paths)
        
        used_fields = [field for field in all_datasets[0]._fields if getattr(all_datasets[0], field) is not None]
        
        assert len(all_datasets) >= 1
        if kwargs.get('sentence_batching'):
            for field in used_fields:
                lengths = list(map(lambda dataset: len(getattr(dataset, field)), all_datasets))
                assert len(set(lengths)) == 1, 'When using sentence batching your datasets should have the same size.'
            if 'train' in used_fields:
                ids_sets = list(map(lambda dataset: set(self.get_train_processed_ids(dataset.train)), all_datasets))
                id_set_base = set(ids_sets[0])
                for id_set in ids_sets:
                    assert set(
                        id_set) == id_set_base, 'When using sentence batching your datasets should have matching ids'
            
            sort_key_fn = process_id
            batch_size_fn = default_batch_fn
        else:
            # use default values for `sort_key_fn` and `batch_size_fn`
            sort_key_fn = input_then_output_len
            batch_size_fn = input_tokens_fn
        
        groups = len(all_datasets) if kwargs.get('sentence_batching') else None
        
        if kwargs.get('separate_eval') and (all_datasets[0].eval or all_datasets[0].test):
            return all_datasets, all_paths
        # TODO fix handling paths for multilingual
        else:
            return self.combine_datasets(all_datasets, all_paths, sort_key_fn, batch_size_fn, used_fields, groups), all_paths[0]


@register_task('almond_multilingual')
class AlmondMultiLingual(BaseAlmondMultiLingualTask):
    def __init__(self, name, args):
        super().__init__(name, args)
        self.lang_as_question = args.almond_lang_as_question
    
    def _is_program_field(self, field_name):
        return field_name == 'answer'
    
    def is_contextual(self):
        return False
    
    @property
    def metrics(self):
        return ['em', 'sm', 'bleu']
    
    def _make_example(self, parts, dir_name, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, sentence, target_code = parts[:3]
        else:
            example_id, sentence, target_code = parts
        language = ISO_to_LANG.get(dir_name, 'English').lower()
        if self.lang_as_question:
            question = 'translate from {} to thingtalk'.format(language)
        else:
            question = 'translate from english to thingtalk'
        context = sentence
        answer = target_code
        return Example.from_raw(self.name + '/' + dir_name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)


@register_task('almond_dialogue_multilingual_nlu')
class AlmondDialogMultiLingualNLU(BaseAlmondMultiLingualTask):
    """Multi-turn NLU task (user and agent) for MultiLingual Almond dialogues
    """
    
    def _is_program_field(self, field_name):
        return field_name in ('answer', 'context')
    
    def is_contextual(self):
        return True
    
    @property
    def metrics(self):
        return ['em', 'sm', 'bleu']
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        if self._almond_has_multiple_programs:
            example_id, context, sentence, target_code = parts
        else:
            example_id, context, sentence, target_code = parts[:4]
        answer = target_code
        question = sentence
        return Example.from_raw(self.name + '/' + dir_name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)


@register_task('almond_dialogue_multilingual_nlg')
class AlmondDialogMultiLingualNLG(BaseAlmondTask):
    """Multi-turn NLG task (agent) for MultiLingual Almond dialogues
    """
    
    def _is_program_field(self, field_name):
        return field_name == 'context'
    
    def is_contextual(self):
        return True
    
    @property
    def metrics(self):
        return ['bleu']
    
    def _make_example(self, parts, dir_name=None, **kwargs):
        # the question is irrelevant for this task
        example_id, context, sentence, target_code = parts
        question = 'what should the agent say ?'
        context = context + ' ' + target_code
        answer = sentence
        return Example.from_raw(self.name + '/' + dir_name + '/' + example_id, context, question, answer,
                                preprocess=self.preprocess_field, lower=False)

