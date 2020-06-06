from .run_pplm_discrim_train import Discriminator
from .pplm_classification_head import ClassificationHead
from .GPT2Seq2Seq import GPT2Seq2Seq
import logging
from typing import List
from transformers import GPT2LMHeadModel
import torch

class GPT2Seq2SeqWithSentiment(GPT2Seq2Seq):
    def __init__(self, config):
        super().__init__(config)

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        sep_token_position = (input_ids==self.sep_token_id).to(torch.long)
        assert (torch.sum(sep_token_position, dim=1)<=1).all(), 'All input_ids must contain zero or one sep_token. sep_token_position = %s\nsep_token_id = %d' % (str(sep_token_position), self.sep_token_id)
        token_type_ids = torch.cumsum(sep_token_position, dim=1) - sep_token_position
        attention_mask = (input_ids!=self.pad_token_id).to(torch.long) # 0 means mask, 1 means no mask
        position_ids = ((torch.cumsum(attention_mask, dim=1)-1)*(1-token_type_ids)+(torch.cumsum(token_type_ids, dim=1)-1)*token_type_ids).clamp(min=0)
        token_type_ids = self.sep_token_id * (1-token_type_ids) + self.end_token_id * token_type_ids

        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        inputs = {"input_ids": input_ids, "position_ids": position_ids, "token_type_ids": token_type_ids, "past": past}
        print({"input_ids": input_ids, "position_ids": position_ids, "token_type_ids": token_type_ids})
        return inputs


    # The following forward function is modified from transformers GPT2LMHeadModel forward fuction
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=True
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
                Language modeling loss.
            prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
                Contains pre-computed hidden-states (key and values in the attention blocks).
                Can be used (see `past` input) to speed up sequential decoding.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.

        Examples:

            import torch
            from transformers import GPT2Tokenizer, GPT2LMHeadModel

            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]

        """

        # if past is not None and (input_ids is None or input_ids.shape[1] == 1):
        inputs = self.prepare_inputs_for_generation(input_ids, past=past)
        if inputs_embeds is not None:
            print('no prep')
            print('inputs_embeds = ', inputs_embeds)
            inputs['input_ids'] = None
            
        transformer_outputs = self.transformer(
            # input_ids,
            inputs['input_ids'], # this may be modified by prepare_inputs_for_generation
            # past=past,
            past=inputs['past'],
            # attention_mask=attention_mask,
            # attention_mask=inputs['attention_mask'],
            # token_type_ids=token_type_ids,
            token_type_ids=inputs['token_type_ids'],
            # position_ids=position_ids,
            position_ids=inputs['position_ids'],
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


# '''
# the following generate_text_pplm function is modified from generate_text_pplm in run_pplm.py
# '''
# def generate_text_pplm(
# 	model,
# 	tokenizer,
# 	context=None,	
# 	logits=None, # context ignored if logits present
# 	past=None,
# 	device="cuda",
# 	perturb=True,
# 	bow_indices=None,
# 	classifier=self.pretrained_sentiment_head,
# 	class_label=2, # positive, 3 for negative
# 	loss_type=0,
# 	length=100,
# 	stepsize=0.02,
# 	temperature=1.0,
# 	top_k=10,
# 	sample=False,
# 	num_iterations=3,
# 	grad_length=10000,
# 	horizon_length=1,
# 	window_length=0,
# 	decay=False,
# 	gamma=1.5,
# 	gm_scale=0.9,
# 	kl_scale=0.01,
# 	repetition_penalty=1.0,
# ):
# 	output_so_far = None
# 	if context:
# 			context_t = torch.tensor(context, device=device, dtype=torch.long)
# 			while len(context_t.shape) < 2:
# 					context_t = context_t.unsqueeze(0)
# 			output_so_far = context_t

# 	# collect one hot vectors for bags of words
# 	one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer, device)

# 	grad_norms = None
# 	last = None
# 	unpert_discrim_loss = 0
# 	loss_in_time = []
# 	for i in trange(length, ascii=True):

# 			# Get past/probs for current output, except for last word
# 			# Note that GPT takes 2 inputs: past + current_token

# 			# run model forward to obtain unperturbed
# 			if past is None and output_so_far is not None:
# 					last = output_so_far[:, -1:]
# 					if output_so_far.shape[1] > 1:
# 							_, past, _ = model(output_so_far[:, :-1])

# 			unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
# 			unpert_last_hidden = unpert_all_hidden[-1]

# 			# check if we are abowe grad max length
# 			if i >= grad_length:
# 					current_stepsize = stepsize * 0
# 			else:
# 					current_stepsize = stepsize

# 			# modify the past if necessary
# 			if not perturb or num_iterations == 0:
# 					pert_past = past

# 			else:
# 					accumulated_hidden = unpert_last_hidden[:, :-1, :]
# 					accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

# 					if past is not None:
# 							pert_past, _, grad_norms, loss_this_iter = perturb_past(
# 									past,
# 									model,
# 									last,
# 									unpert_past=unpert_past,
# 									unpert_logits=unpert_logits,
# 									accumulated_hidden=accumulated_hidden,
# 									grad_norms=grad_norms,
# 									stepsize=current_stepsize,
# 									one_hot_bows_vectors=one_hot_bows_vectors,
# 									classifier=classifier,
# 									class_label=class_label,
# 									loss_type=loss_type,
# 									num_iterations=num_iterations,
# 									horizon_length=horizon_length,
# 									window_length=window_length,
# 									decay=decay,
# 									gamma=gamma,
# 									kl_scale=kl_scale,
# 									device=device,
# 							)
# 							loss_in_time.append(loss_this_iter)
# 					else:
# 							pert_past = past

# 			pert_logits, past, pert_all_hidden = model(last, past=pert_past)
# 			pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST

# 			for token_idx in set(output_so_far[0].tolist()):
# 					if pert_logits[0, token_idx] < 0:
# 							pert_logits[0, token_idx] *= repetition_penalty
# 					else:
# 							pert_logits[0, token_idx] /= repetition_penalty

# 			pert_probs = F.softmax(pert_logits, dim=-1)

# 			if classifier is not None:
# 					ce_loss = torch.nn.CrossEntropyLoss()
# 					prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
# 					label = torch.tensor([class_label], device=device, dtype=torch.long)
# 					unpert_discrim_loss = ce_loss(prediction, label)
# 					print("unperturbed discrim loss", unpert_discrim_loss.data.cpu().numpy())
# 			else:
# 					unpert_discrim_loss = 0

# 			# Fuse the modified model and original model
# 			if perturb:

# 					unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

# 					pert_probs = (pert_probs ** gm_scale) * (unpert_probs ** (1 - gm_scale))  # + SMALL_CONST
# 					pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST

# 					# rescale
# 					if torch.sum(pert_probs) <= 1:
# 							pert_probs = pert_probs / torch.sum(pert_probs)

# 			else:
# 					pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
# 					pert_probs = F.softmax(pert_logits, dim=-1)

# 			# sample or greedy
# 			if sample:
# 					last = torch.multinomial(pert_probs, num_samples=1)

# 			else:
# 					_, last = torch.topk(pert_probs, k=1, dim=-1)

# 			# update context/output_so_far appending the new token
# 			output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)

# 			print(tokenizer.decode(output_so_far.tolist()[0]))

# 	return output_so_far, unpert_discrim_loss, loss_in_time





# '''
# main() taken from run_pplm.py
# '''
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--pretrained_model",
#         "-M",
#         type=str,
#         default="gpt2-medium",
#         help="pretrained model name or path to local checkpoint",
#     )
#     parser.add_argument("--cond_text", type=str, default="The lake", help="Prefix texts to condition on")
#     parser.add_argument("--uncond", action="store_true", help="Generate from end-of-text as prefix")
#     parser.add_argument(
#         "--num_samples", type=int, default=1, help="Number of samples to generate from the modified latents",
#     )
#     parser.add_argument(
#         "--bag_of_words",
#         "-B",
#         type=str,
#         default=None,
#         help="Bags of words used for PPLM-BoW. "
#         "Either a BOW id (see list in code) or a filepath. "
#         "Multiple BoWs separated by ;",
#     )
#     parser.add_argument(
#         "--discrim",
#         "-D",
#         type=str,
#         default=None,
#         choices=("clickbait", "sentiment", "toxicity", "generic"),
#         help="Discriminator to use",
#     )
#     parser.add_argument("--discrim_weights", type=str, default=None, help="Weights for the generic discriminator")
#     parser.add_argument(
#         "--discrim_meta", type=str, default=None, help="Meta information for the generic discriminator"
#     )
#     parser.add_argument(
#         "--class_label", type=int, default=-1, help="Class label used for the discriminator",
#     )
#     parser.add_argument("--length", type=int, default=100)
#     parser.add_argument("--stepsize", type=float, default=0.02)
#     parser.add_argument("--temperature", type=float, default=1.0)
#     parser.add_argument("--top_k", type=int, default=10)
#     parser.add_argument("--sample", action="store_true", help="Generate from end-of-text as prefix")
#     parser.add_argument("--num_iterations", type=int, default=3)
#     parser.add_argument("--grad_length", type=int, default=10000)
#     parser.add_argument(
#         "--window_length",
#         type=int,
#         default=0,
#         help="Length of past which is being optimized; " "0 corresponds to infinite window length",
#     )
#     parser.add_argument(
#         "--horizon_length", type=int, default=1, help="Length of future to optimize over",
#     )
#     parser.add_argument("--decay", action="store_true", help="whether to decay or not")
#     parser.add_argument("--gamma", type=float, default=1.5)
#     parser.add_argument("--gm_scale", type=float, default=0.9)
#     parser.add_argument("--kl_scale", type=float, default=0.01)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--no_cuda", action="store_true", help="no cuda")
#     parser.add_argument("--colorama", action="store_true", help="colors keywords")
#     parser.add_argument(
#         "--repetition_penalty", type=float, default=1.0, help="Penalize repetition. More than 1.0 -> less repetition",
#     )

#     args = parser.parse_args()
#     run_pplm_example(**vars(args))
