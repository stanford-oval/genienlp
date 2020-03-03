#
# Copyright (c) 2018, Salesforce, Inc.
#                     The Board of Trustees of the Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
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

import math
import torch
from torch import nn
from torch.nn import functional as F

from .common import CombinedEmbedding, TransformerDecoder, LSTMDecoderAttention, Feedforward, \
    mask, positional_encodings_like, EPSILON, MultiLSTMCell, BeamHypotheses
from ..util import top_k_top_p_filtering


class MQANDecoder(nn.Module):
    def __init__(self, numericalizer, args, decoder_embeddings):
        super().__init__()
        self.numericalizer = numericalizer
        self.pad_idx = numericalizer.pad_id
        self.init_idx = numericalizer.init_id
        self.args = args

        self.decoder_embeddings = CombinedEmbedding(numericalizer, decoder_embeddings, args.dimension,
                                                    trained_dimension=args.trainable_decoder_embeddings,
                                                    project=True,
                                                    finetune_pretrained=False)

        if args.transformer_layers > 0:
            self.self_attentive_decoder = TransformerDecoder(args.dimension, args.transformer_heads,
                                                             args.transformer_hidden,
                                                             args.transformer_layers,
                                                             args.dropout_ratio)
        else:
            self.self_attentive_decoder = None

        if args.rnn_layers > 0:
            self.rnn_decoder = LSTMDecoder(args.dimension, args.rnn_dimension,
                                           dropout=args.dropout_ratio, num_layers=args.rnn_layers)
            switch_input_len = 2 * args.rnn_dimension + args.dimension
        else:
            self.context_attn = LSTMDecoderAttention(args.dimension, dot=True)
            self.question_attn = LSTMDecoderAttention(args.dimension, dot=True)
            self.dropout = nn.Dropout(args.dropout_ratio)
            switch_input_len = 2 * args.dimension
        self.vocab_pointer_switch = nn.Sequential(Feedforward(switch_input_len, 1), nn.Sigmoid())
        self.context_question_switch = nn.Sequential(Feedforward(switch_input_len, 1), nn.Sigmoid())

        self.generative_vocab_size = numericalizer.generative_vocab_size
        self.out = nn.Linear(args.rnn_dimension if args.rnn_layers > 0 else args.dimension, self.generative_vocab_size)

    def set_embeddings(self, embeddings):
        if self.decoder_embeddings is not None:
            self.decoder_embeddings.set_embeddings(embeddings)

    def forward(self, batch, self_attended_context, final_context, context_rnn_state, final_question,
                question_rnn_state):
        context, context_lengths, context_limited = batch.context.value, batch.context.length, batch.context.limited
        question, question_lengths, question_limited = batch.question.value, batch.question.length, batch.question.limited
        answer, answer_lengths, answer_limited = batch.answer.value, batch.answer.length, batch.answer.limited
        decoder_vocab = batch.decoder_vocab

        self.map_to_full = decoder_vocab.decode

        context_padding = context.data == self.pad_idx
        question_padding = question.data == self.pad_idx

        if self.training:
            if self.args.rnn_layers > 0:
                self.rnn_decoder.applyMasks(context_padding, question_padding)
            else:
                self.context_attn.applyMasks(context_padding)
                self.question_attn.applyMasks(question_padding)

            answer_padding = (answer.data == self.pad_idx)[:, :-1]

            answer_embedded = self.decoder_embeddings(answer[:, :-1], padding=answer_padding).last_layer

            if self.args.transformer_layers > 0:
                self_attended_decoded = self.self_attentive_decoder(answer_embedded,
                                                                    self_attended_context,
                                                                    context_padding=context_padding,
                                                                    answer_padding=answer_padding,
                                                                    positional_encodings=True)
            else:
                self_attended_decoded = answer_embedded

            if self.args.rnn_layers > 0:
                rnn_decoder_outputs = self.rnn_decoder(self_attended_decoded, final_context, final_question,
                                                       hidden=context_rnn_state)
                decoder_output, vocab_pointer_switch_input, context_question_switch_input, context_attention, \
                question_attention, rnn_state = rnn_decoder_outputs
            else:
                context_decoder_output, context_attention = self.context_attn(self_attended_decoded, final_context)
                question_decoder_output, question_attention = self.question_attn(self_attended_decoded, final_question)

                vocab_pointer_switch_input = torch.cat((context_decoder_output, self_attended_decoded), dim=-1)
                context_question_switch_input = torch.cat((question_decoder_output, self_attended_decoded), dim=-1)

                decoder_output = self.dropout(context_decoder_output)

            vocab_pointer_switch = self.vocab_pointer_switch(vocab_pointer_switch_input)
            context_question_switch = self.context_question_switch(context_question_switch_input)

            probs = self.probs(decoder_output, vocab_pointer_switch, context_question_switch,
                               context_attention, question_attention,
                               context_limited, question_limited,
                               decoder_vocab)

            probs, targets = mask(answer_limited[:, 1:].contiguous(), probs.contiguous(), pad_idx=decoder_vocab.pad_idx)
            loss = F.nll_loss(probs.log(), targets)
            return loss, None

        else:
            return None, self.decode(self_attended_context, final_context, context_padding, final_question, question_padding,
                                     context_limited, question_limited,
                                     decoder_vocab, rnn_state=context_rnn_state).data

    def probs(self, outputs, vocab_pointer_switches, context_question_switches,
              context_attention, question_attention,
              context_indices, question_indices,
              decoder_vocab):

        size = list(outputs.size())

        size[-1] = self.generative_vocab_size
        scores = self.out(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim() - 1)
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = len(decoder_vocab)
        if self.generative_vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = scaled_p_vocab.new_full(size, EPSILON)
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim() - 1)

        # p_context_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim() - 1, context_indices.unsqueeze(1).expand_as(context_attention),
                                    (context_question_switches * (1 - vocab_pointer_switches)).expand_as(
                                        context_attention) * context_attention)

        # p_question_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim() - 1,
                                    question_indices.unsqueeze(1).expand_as(question_attention),
                                    ((1 - context_question_switches) * (1 - vocab_pointer_switches)).expand_as(
                                        question_attention) * question_attention)

        return scaled_p_vocab

    def decode(self, self_attended_context, context, context_padding, question, question_padding, context_indices, question_indices,
               decoder_vocab, rnn_state=None):
        batch_size = context.size()[0]
        max_decoder_time = self.args.max_output_length

        input_ids = self_attended_context[-1].new_full((batch_size, 1), self.init_idx, dtype=torch.long)


        decoder_wrapper = MQANDecoderWrapper(self_attended_context, context, context_padding, question, question_padding, context_indices, question_indices,
               decoder_vocab, rnn_state, batch_size, max_decoder_time, self, num_beams=self.args.num_beams)
        
        if self.args.num_beams > 1:
            outputs = self._decode_beam_search(
                input_ids=input_ids,
                max_length=self.args.max_output_length,
                do_sample=False,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=decoder_vocab.pad_idx,
                eos_token_ids=[decoder_vocab.eos_idx],
                batch_size=batch_size,
                length_penalty=1.0,
                num_beams=self.args.num_beams,
                vocab_size=len(decoder_vocab),
                decoder_wrapper=decoder_wrapper,
            )
        else:
            outputs = self._decode_greedy(
                input_ids=input_ids,
                max_length=self.args.max_output_length,
                pad_token_id=decoder_vocab.pad_idx,
                eos_token_id=decoder_vocab.eos_idx,
                batch_size=batch_size,
                decoder_wrapper=decoder_wrapper,
            )

        # print('outputs = ', outputs.shape)
        return outputs

    def _decode_greedy(
        self,
        input_ids,
        max_length,
        pad_token_id,
        eos_token_id,
        batch_size,
        decoder_wrapper
    ):
        
        outs = input_ids.new_full((batch_size, max_length), pad_token_id, dtype=torch.long)
        eos_yet = input_ids.new_zeros((batch_size,)).byte()
        for t in range(max_length):
            probs = decoder_wrapper.next_token_probs(input_ids[:, -1].unsqueeze(-1))
            pred_probs, preds = probs.max(-1)
            eos_yet = eos_yet | (preds == eos_token_id).byte()
            outs[:, t] = preds.cpu().apply_(self.map_to_full)
            if eos_yet.all():
                break
            input_ids = torch.cat((input_ids, outs[:, t].unsqueeze(1)), dim=1)
        return outs

    def _decode_beam_search(
        self,
        input_ids,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
        decoder_wrapper,
    ):
        """
        Generate sequences for each example with beam search. Temperature, repetition penalty, top-k and top-p sampling are also supported.
        """

        cur_len = 1
        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, 1)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, 1)  # (batch_size * num_beams, cur_len)

        # generated hypotheses
        generated_hyps = [BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            # next_token_probs outputs a normalized probability distribution instead of logits
            scores = torch.log(decoder_wrapper.next_token_probs(input_ids[:, -1].unsqueeze(-1)))  # (batch_size * num_beams, vocab_size)

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2)  # (batch_size * num_beams, vocab_size)
                # Sample 2 next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1), num_samples=2)  # (batch_size * num_beams, 2)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, 2)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, 2)
                # Match shape of greedy beam search
                next_words = next_words.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
                next_scores = next_scores.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):

                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):
                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    # print('eos_token_ids = ', eos_token_ids)
                    # print('word_id.item() = ', word_id.item())
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item())
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                assert len(next_sent_beam) == (0 if cur_len + 1 == max_length else num_beams)
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            device = beam_words.device
            beam_words = beam_words.cpu().apply_(self.map_to_full).to(device)
            input_ids = input_ids[beam_idx, :]
            decoder_wrapper.reorder(beam_idx)
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # select the best hypotheses
        tgt_len = input_ids.new(batch_size)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = input_ids.new(batch_size, tgt_len.max().item()).fill_(pad_token_id)
        for i, hypo in enumerate(best):
            decoded[i, : tgt_len[i] - 1] = hypo
            decoded[i, tgt_len[i] - 1] = self.map_to_full(eos_token_ids[0])

        return decoded


class LSTMDecoder(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.0, num_layers=1):
        super().__init__()
        self.d_hid = d_hid
        self.d_in = d_in
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.input_feed = True
        if self.input_feed:
            d_in += 1 * d_hid

        self.rnn = MultiLSTMCell(self.num_layers, d_in, d_hid, dropout)
        self.context_attn = LSTMDecoderAttention(d_hid, dot=True)
        self.question_attn = LSTMDecoderAttention(d_hid, dot=True)

    def applyMasks(self, context_mask, question_mask):
        self.context_attn.applyMasks(context_mask)
        self.question_attn.applyMasks(question_mask)

    def forward(self, input: torch.Tensor, context, question, output=None, hidden=None):
        context_output = output if output is not None else self.make_init_output(context)

        context_outputs, vocab_pointer_switch_inputs, context_question_switch_inputs, context_attentions, question_attentions = [], [], [], [], []
        for decoder_input in input.split(1, dim=1):
            context_output = self.dropout(context_output)
            if self.input_feed:
                rnn_input = torch.cat([decoder_input, context_output], 2)
            else:
                rnn_input = decoder_input

            rnn_input = rnn_input.squeeze(1)
            dec_state, hidden = self.rnn(rnn_input, hidden)
            dec_state = dec_state.unsqueeze(1)

            # print('dec_state = ', dec_state.shape)
            # print('context = ', context.shape)
            context_output, context_attention = self.context_attn(dec_state, context)
            question_output, question_attention = self.question_attn(dec_state, question)
            vocab_pointer_switch_inputs.append(torch.cat([dec_state, context_output, decoder_input], -1))
            context_question_switch_inputs.append(torch.cat([dec_state, question_output, decoder_input], -1))

            context_output = self.dropout(context_output)
            context_outputs.append(context_output)
            context_attentions.append(context_attention)
            question_attentions.append(question_attention)

        return [torch.cat(x, dim=1) for x in (context_outputs,
                                              vocab_pointer_switch_inputs,
                                              context_question_switch_inputs,
                                              context_attentions,
                                              question_attentions)] + [hidden]

    def make_init_output(self, context):
        batch_size = context.size(0)
        h_size = (batch_size, 1, self.d_hid)
        return context.new_zeros(h_size)


class MQANDecoderWrapper(object):
    """
    A wrapper for MQANDecoder that wraps around its recurrent neural network, so that we can decode it like a Transformer
    """

    def __init__(self, self_attended_context, context, context_padding, question, question_padding, context_indices, question_indices,
               decoder_vocab, rnn_state, batch_size, max_decoder_time, mqan_decoder: MQANDecoder, num_beams:int):
        # print('self_attended_context = ', self_attended_context)
        self.decoder_vocab = decoder_vocab
        # if num_beams > 1:
        self_attended_context = self.expand_for_beam_search(self_attended_context, batch_size, num_beams)
        context = self.expand_for_beam_search(context, batch_size, num_beams)
        context_padding = self.expand_for_beam_search(context_padding, batch_size, num_beams)
        question = self.expand_for_beam_search(question, batch_size, num_beams)
        question_padding = self.expand_for_beam_search(question_padding, batch_size, num_beams)
        context_indices = self.expand_for_beam_search(context_indices, batch_size, num_beams)
        question_indices = self.expand_for_beam_search(question_indices, batch_size, num_beams)
        rnn_state = self.expand_for_beam_search(rnn_state, batch_size, num_beams, dim=1)
        self.self_attended_context = self_attended_context
        self.context = context
        self.context_padding = context_padding
        self.question = question
        self.question_padding = question_padding
        self.context_indices = context_indices
        self.question_indices = question_indices
        self.rnn_state = rnn_state
        self.batch_size = batch_size
        self.max_decoder_time = max_decoder_time
        self.mqan_decoder = mqan_decoder
        self.num_beams = num_beams

        if self.mqan_decoder.args.rnn_layers > 0:
                self.mqan_decoder.rnn_decoder.applyMasks(self.context_padding, self.question_padding)
        else:
            self.mqan_decoder.context_attn.applyMasks(self.context_padding)
            self.mqan_decoder.question_attn.applyMasks(self.question_padding)

        self.time = 0
        self.decoder_output = None

        if self.mqan_decoder.args.transformer_layers > 0:
            self.hiddens = [self.self_attended_context[0].new_zeros((self.batch_size*self.num_beams, self.max_decoder_time, self.mqan_decoder.args.dimension))
                    for l in range(len(self.mqan_decoder.self_attentive_decoder.layers) + 1)]
            self.hiddens[0] =  self.hiddens[0] + positional_encodings_like(self.hiddens[0])

    
    def reorder(self, new_order):
        # TODO only reordering rnn_state should be enough since reordering happens among beams of the same input
        self.self_attended_context = self.reorder_for_beam_search(self.self_attended_context, new_order)
        self.context = self.reorder_for_beam_search(self.context, new_order)
        self.context_padding = self.reorder_for_beam_search(self.context_padding, new_order)
        self.question = self.reorder_for_beam_search(self.question, new_order)
        self.question_padding = self.reorder_for_beam_search(self.question_padding, new_order)
        self.context_indices = self.reorder_for_beam_search(self.context_indices, new_order)
        self.question_indices = self.reorder_for_beam_search(self.question_indices, new_order)
        self.rnn_state = self.reorder_for_beam_search(self.rnn_state, new_order, dim=1)

        if self.mqan_decoder.args.rnn_layers > 0:
                self.mqan_decoder.rnn_decoder.applyMasks(self.context_padding, self.question_padding)
        else:
            self.mqan_decoder.context_attn.applyMasks(self.context_padding)
            self.mqan_decoder.question_attn.applyMasks(self.question_padding)


    def next_token_probs(self, current_token_id):
        # print('current_`token_id = ', current_token_id, current_token_id.shape)
        embedding = self.mqan_decoder.decoder_embeddings(current_token_id).last_layer

        if self.mqan_decoder.args.transformer_layers > 0:
            self.hiddens[0][:, self.time] = self.hiddens[0][:, self.time] + \
                                (math.sqrt(self.mqan_decoder.self_attentive_decoder.d_model) * embedding).squeeze(1)
            for l in range(len(self.mqan_decoder.self_attentive_decoder.layers)):
                self.hiddens[l + 1][:, self.time] = self.mqan_decoder.self_attentive_decoder.layers[l](self.hiddens[l][:, self.time],
                                                                                self.self_attended_context[l],
                                                                                selfattn_keys=self.hiddens[l][:, :self.time + 1],
                                                                                context_padding=self.context_padding)

            self_attended_decoded = self.hiddens[-1][:, self.time].unsqueeze(1)
        else:
            self_attended_decoded = embedding

        if self.mqan_decoder.args.rnn_layers > 0:
            # print(self_attended_decoded.shape)
            # print(self.context.shape)
            # print(self.question.shape)
            # print(self.rnn_state[0].shape, self.rnn_state[1].shape)
            # print(self.decoder_output.shape)
            rnn_decoder_outputs = self.mqan_decoder.rnn_decoder(self_attended_decoded, self.context, self.question,
                                                    hidden=self.rnn_state, output=self.decoder_output)
            self.decoder_output, vocab_pointer_switch_input, context_question_switch_input, context_attention, \
                question_attention, self.rnn_state = rnn_decoder_outputs
        else:
            context_decoder_output, context_attention = self.mqan_decoder.context_attn(self_attended_decoded, self.context)
            question_decoder_output, question_attention = self.mqan_decoder.question_attn(self_attended_decoded, self.question)

            vocab_pointer_switch_input = torch.cat((context_decoder_output, self_attended_decoded), dim=-1)
            context_question_switch_input = torch.cat((question_decoder_output, self_attended_decoded), dim=-1)

            self.decoder_output = self.dropout(context_decoder_output)

        vocab_pointer_switch = self.mqan_decoder.vocab_pointer_switch(vocab_pointer_switch_input)
        context_question_switch = self.mqan_decoder.context_question_switch(context_question_switch_input)

        probs = self.mqan_decoder.probs(self.decoder_output, vocab_pointer_switch, context_question_switch,
                            context_attention, question_attention,
                            self.context_indices, self.question_indices, self.decoder_vocab)

        self.time += 1
        return probs.squeeze(1)

    def expand_for_beam_search(self, t, batch_size, num_beams, dim=0):
        if isinstance(t, tuple):
            elements = []
            for e in t:
                elements.append(self.expand_for_beam_search(e, batch_size, num_beams, dim))
            return tuple(elements)
        elif isinstance(t, list):
            elements = []
            for e in t:
                elements.append(self.expand_for_beam_search(e, batch_size, num_beams, dim))
            return elements

        # print('before expansion: ', t.shape)
        original_size = list(t.shape)
        original_size[dim] *= num_beams
        t = t.unsqueeze(dim+1)
        expanded_size = list(t.shape)
        expanded_size[dim+1] = num_beams
        t = t.expand(*expanded_size)
        t = t.contiguous().view(*original_size)  # (batch_size * num_beams, -1)
        # print('after expansion: ', t.shape)
        return t

    def reorder_for_beam_search(self, t, new_order, dim=0):
        if isinstance(t, tuple):
            elements = []
            for e in t:
                elements.append(self.reorder_for_beam_search(e, new_order, dim))
            return tuple(elements)
        elif isinstance(t, list):
            elements = []
            for e in t:
                elements.append(self.reorder_for_beam_search(e, new_order, dim))
            return elements

        # print('before reordering t = ', t)
        p = [i for i in range(len(t.shape))]
        p[dim] = 0
        p[0] = dim
        t = t.permute(*p)
        t = t[new_order]
        t = t.permute(*p)
        # print('after reordering t = ', t)

        return t
