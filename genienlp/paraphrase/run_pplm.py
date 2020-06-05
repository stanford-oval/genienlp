#! /usr/bin/env python3
# coding=utf-8

# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
python run_pplm.py --discriminator sentiment --class_label 3 --input_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange

from genienlp.paraphrase.pplm_classification_head import ClassificationHead
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel
from genienlp.paraphrase.GPT2Seq2SeqWithSentiment import GPT2Seq2SeqWithSentiment


PPLM_DISCRIM = 2
SMALL_CONST = 1e-15
BIG_CONST = 1e10

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}

logger = logging.getLogger(__name__)


def to_var(x, requires_grad=False, volatile=False, device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        x = x.cuda()
    elif device != "cuda":
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)


def perturb_past(
    past,
    model,
    last,
    unpert_past=None,
    unpert_logits=None,
    accumulated_hidden=None,
    grad_norms=None,
    stepsize=0.01,
    classifier=None,
    class_label=None,
    loss_type=0,
    num_iterations=3,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    kl_scale=0.01,
    device="cuda",
):
    # Generate inital perturbed past
    grad_accumulator = [(np.zeros(p.shape).astype("float32")) for p in past]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.0, 1.0 + SMALL_CONST, 1.0 / (window_length))[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(
            past[0].shape[:-2]) + tuple([window_length]) + tuple(past[0].shape[-1:])

        zeros_key_val_shape = (
            tuple(past[0].shape[:-2]) + tuple([curr_length -
                                               window_length]) + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(
            zeros_key_val_shape)), dim=-2).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in trange(num_iterations):
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device) for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + \
            torch.sum(hidden, dim=1).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []

        if loss_type == 2 or loss_type == 3:
            ce_loss = torch.nn.CrossEntropyLoss()
            # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                _, curr_unpert_past, curr_all_hidden = model(
                    past=curr_unpert_past, inputs_embeds=inputs_embeds)
                curr_hidden = curr_all_hidden[-1]
                new_accumulated_hidden = new_accumulated_hidden + \
                    torch.sum(curr_hidden, dim=1)

            prediction = classifier(new_accumulated_hidden / (curr_length + 1 + horizon_length))

            label = torch.tensor(prediction.shape[0] * [class_label], device=device, dtype=torch.long)
            discriminator_loss = ce_loss(prediction, label)
            logging.debug("pplm_discriminator_loss:", discriminator_loss.data.cpu().numpy())
            loss += discriminator_loss
            loss_list.append(discriminator_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = unpert_probs + SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(device).detach()
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * ((corrected_probs * (corrected_probs / unpert_probs).log()).sum())
            logging.debug("kl_loss %f", kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        logging.debug("pplm_loss %f", (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        grad_norms = [
            (torch.norm(p_.grad * window_mask) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation)
        ]

        # normalize gradients
        grad = [
            -stepsize * (p_.grad * window_mask /
                         grad_norms[index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [to_var(torch.from_numpy(
        p_), requires_grad=True, device=device) for p_ in grad_accumulator]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(name: Optional[str], class_label: Union[str, int], device: str) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params["class_size"], embed_size=params["embed_size"]).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError(
            "Either url or path have to be specified " "in the discriminator model parameters")
    classifier.load_state_dict(torch.load(
        resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            print("class_label {} not in class_vocab".format(class_label))
            print("available values are: {}".format(params["class_vocab"]))
            print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def full_text_generation(
    model,
    tokenizer,
    context=None,
    num_samples=1,
    device="cuda",
    discriminator=None,
    class_label=None,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    repetition_penalty=1.0,
    **kwargs
):
    classifier, class_id = get_classifier(discriminator, class_label, device)

    if classifier is not None:
        loss_type = PPLM_DISCRIM
    else:
        raise Exception("Specify either a discriminator")

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        repetition_penalty=repetition_penalty,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discriminator_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discriminator_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            repetition_penalty=repetition_penalty,
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discriminator_losses.append(discriminator_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == "cuda":
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discriminator_losses, losses_in_time


def generate_text_pplm(
    model,
    tokenizer,
    context=None,
    past=None,
    device="cuda",
    perturb=True,
    classifier=None,
    class_label=None,
    loss_type=0,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    repetition_penalty=1.0,
):
    output_so_far = None
    sep_token_position = None
    if context:
        sep_token_position = len(context)
        # context_t = torch.tensor(context, device=device, dtype=torch.long)

        # line below may not be necessary if context is never more than 1 example
        context_t = torch.tensor(model.pad_to_max_length([context]), dtype=torch.long, device=device)
        # attention_mask = None
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t


    grad_norms = None
    last = None
    unpert_discriminator_loss = 0
    loss_in_time = []
    for i in trange(length, ascii=True):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] - sep_token_position > 1:
                _, past, _ = model(output_so_far[:, :-1])
        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)

        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are above grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST

        for token_idx in set(output_so_far[0].tolist()):
            if pert_logits[0, token_idx] < 0:
                pert_logits[0, token_idx] *= repetition_penalty
            else:
                pert_logits[0, token_idx] /= repetition_penalty

        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device, dtype=torch.long)
            unpert_discriminator_loss = ce_loss(prediction, label)
            print("unperturbed discriminator loss", unpert_discriminator_loss.data.cpu().numpy())
        else:
            unpert_discriminator_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = (pert_probs ** gm_scale) * \
                (unpert_probs ** (1 - gm_scale))  # + SMALL_CONST
            pert_probs = top_k_filter(
                pert_probs, k=top_k, probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = last if output_so_far is None else torch.cat(
            (output_so_far, last), dim=1)

        print(tokenizer.decode(output_so_far.tolist()[0]))

    return output_so_far, unpert_discriminator_loss, loss_in_time


def set_generic_model_params(discriminator_weights, discriminator_meta):
    if discriminator_weights is None:
        raise ValueError(
            "When using a generic discriminator, " "discriminator_weights need to be specified")
    if discriminator_meta is None:
        raise ValueError(
            "When using a generic discriminator, " "discriminator_meta need to be specified")

    with open(discriminator_meta, "r") as discriminator_meta_file:
        meta = json.load(discriminator_meta_file)
    meta["path"] = discriminator_weights
    DISCRIMINATOR_MODELS_PARAMS["generic"] = meta


def run_pplm_example(
    pretrained_model,
    input_text="",
    num_samples=1,
    discriminator=None,
    discriminator_weights=None,
    discriminator_meta=None,
    class_label=-1,
    length=100,
    stepsize=0.02,
    temperature=1.0,
    top_k=10,
    sample=False,
    num_iterations=3,
    grad_length=10000,
    horizon_length=1,
    window_length=0,
    decay=False,
    seed=0,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    no_cuda=False,
    repetition_penalty=1.0,
):
    # load pretrained model

    # essentially how the model is called in run_generation.py
    model_class, tokenizer_class, special_tokens = GPT2Seq2SeqWithSentiment, GPT2Tokenizer, {'sep_token': '<paraphrase>', 'end_token': '</paraphrase>'}

    # load model
    logger.info('Loading pretrained model from %s', pretrained_model)
    model = model_class.from_pretrained(pretrained_model, output_hidden_states=True)
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_model)
    end_token_id = tokenizer.convert_tokens_to_ids(special_tokens['end_token'])
    sep_token_id = tokenizer.convert_tokens_to_ids(special_tokens['sep_token'])
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if pad_token_id is None:
        logger.error('Your tokenizer does not have a padding token')

    model.set_token_ids(end_token_id=end_token_id, 
                        sep_token_id=sep_token_id, 
                        pad_token_id=pad_token_id)

    # freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    tokenized_input_text = tokenizer.encode(input_text + special_tokens['sep_token'])
    logging.info("Tokenized input text = %s", tokenized_input_text)

    # generate unperturbed and perturbed texts

    # full_text_generation returns (unpert_gen_tok_text, pert_gen_tok_texts, discriminator_losses, losses_in_time)
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_input_text,
        device=device,
        num_samples=num_samples,
        discriminator=discriminator,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        repetition_penalty=repetition_penalty,
    )
    # untokenize unperturbed text
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

    logging.info("Unperturbed generated text = %s", unpert_gen_text)

    generated_texts = []

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])
            logging.info("Perturbed generated text %d = %s", i + 1, pert_gen_text)
        except Exception as exc:
            logger.warning("Ignoring error while generating perturbed text: %s", str(exc))

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append((tokenized_input_text, pert_gen_tok_text, unpert_gen_tok_text))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, required=True, help="pretrained model name or path to local checkpoint")
    parser.add_argument("--input_text", type=str, default="The lake", help="Prefix texts to condition on")
    parser.add_argument("--discriminator", type=str,default=None,choices=("clickbait", "sentiment", "toxicity", "generic"),
                        help="Discriminator to use")
    parser.add_argument("--discriminator_weights", type=str, default=None,help="Weights for the generic discriminator")
    parser.add_argument("--discriminator_meta", type=str, default=None, help="Meta information for the generic discriminator")
    parser.add_argument("--class_label", type=int, default=-1, help="Class label used for the discriminator")

    # generation hyperparameters
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate from the modified latents",)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Penalize repetition. More than 1.0 -> less repetition")

    # PPLM's gradient update hyperparameters
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument("--window_length",type=int, default=0, help="Length of past which is being optimized; " "0 corresponds to infinite window length")
    parser.add_argument("--horizon_length", type=int, default=1, help="Length of future to optimize over")
    parser.add_argument("--decay", action="store_true", help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")

    args = parser.parse_args()

    # set the device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # set Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.discriminator == "generic":
        set_generic_model_params(args.discriminator_weights, args.discriminator_meta)

    # TODO overwriting args.pretrained_model is temporarily disabled. Enable again after training the new model is done
    # if args.discriminator is not None:
        # args.pretrained_model = DISCRIMINATOR_MODELS_PARAMS[args.discriminator]["pretrained_model"]
        # logger.info("discriminator = {}, pretrained_model set " "to discriminator's = {}".format(args.discriminator, pretrained_model))

    args.sample = (args.temperature != 0)
    run_pplm_example(**vars(args))
