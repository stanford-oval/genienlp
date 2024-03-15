#
# Copyright (c) 2018, Salesforce, Inc.
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
import copy
import logging
from collections import OrderedDict, defaultdict
from typing import List, Union

import dialogues
import sacrebleu
import evaluate

from .util import QUOTED_MATCH_REGEX

logger = logging.getLogger(__name__)

# metrics that are calculated over a corpus (i.e. a list of predictions and gold answers, not single ones).
# These metrics cannot be calculated on individual examples and then averaged.
corpus_level_metrics = {'bleu', 'casedbleu', 'jga'}


def exact_match(prediction, ground_truth):
    return prediction == ground_truth


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for idx, ground_truth in enumerate(ground_truths):
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def computeROUGE(outputs, targets, rouge_types):
    targets = [target[0] for target in targets]
    rouge_metric = evaluate.load('rouge')
    return rouge_metric.compute(references=targets, predictions=outputs, rouge_types=rouge_types)


def computeEM(outputs, targets):
    outs = [metric_max_over_ground_truths(exact_match, o, t) for o, t in zip(outputs, targets)]
    return sum(outs) / len(outputs) * 100


def computeBLEU(outputs, targets):
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return sacrebleu.corpus_bleu(outputs, targets, lowercase=True).score


def computeCasedBLEU(outputs, targets):
    # lowercase is false
    sacrebleu_metric = evaluate.load("sacrebleu")
    return sacrebleu_metric.compute(predictions=outputs, references=targets, lowercase=False)['score']


def compute_e2e_dialogue_score(greedy, answer, args, example_ids, contexts):
    num_examples = len(answer)
    subtask_metrics_dict = OrderedDict()

    new_metrics = [a.upper() + '_' + b for (a, b) in zip(args.e2e_dialogue_valid_subtasks, args.e2e_dialogue_valid_submetrics)]

    results = OrderedDict({'e2e_dialogue_score': 0.0})
    subtask2result_key = OrderedDict({})
    for i, m in enumerate(new_metrics):
        results[m] = 0.0
        subtask2result_key[args.e2e_dialogue_valid_subtasks[i]] = m

    for k, subtask in enumerate(args.e2e_dialogue_valid_subtasks):
        ids, inputs, preds, golds = [], [], [], []
        for i in range(num_examples):
            id_ = example_ids[i]
            if id_.endswith(f'/{subtask}'):
                ids.append(id_)
                inputs.append(contexts[i])
                preds.append(greedy[i])
                golds.append(answer[i])

        if golds:
            metrics_to_compute = args.e2e_dialogue_valid_submetrics[k]
            sub_metrics = compute_metrics(preds, golds, [metrics_to_compute], args, ids, inputs)
            subtask_metrics_dict[subtask] = (
                sub_metrics[metrics_to_compute],
                len(golds),
            )
            
    weighted_num_examples = 0
    for subtask, (sub_result, num_ex) in subtask_metrics_dict.items():
        result_key = subtask2result_key[subtask]

        results[result_key] += sub_result
        results['e2e_dialogue_score'] += (sub_result * num_ex)
        weighted_num_examples += num_ex

    results['e2e_dialogue_score'] /= weighted_num_examples

    return results


def computeSER(greedy, inputs, task):
    dataset_class = getattr(dialogues, task.dataset_name)
    dataset = dataset_class()

    act_values = []
    for input in inputs:
        act_values.append(QUOTED_MATCH_REGEX.findall(input))

    return dataset.compute_ser(greedy, act_values)


def computeDA_EM(greedy, answer, task):
    # Uses dialogues da metric which takes care of entity normalizations

    dataset_class = getattr(dialogues, task.dataset_name)
    dataset = dataset_class()

    answer = [a[0] for a in answer]
    return dataset.compute_da(greedy, answer)


def computeJGA(greedy, answer, example_ids, task):
    # Inputs contain diff states, so we need to compute the full state first
    # TODO: how to address multitask evaluation?
    dataset_class = getattr(dialogues, task.dataset_name)
    dataset = dataset_class()

    cur_dial_id = None
    full_answer = []
    full_greedy = []
    assert len(example_ids) == len(greedy) == len(answer)
    for id_, g, a in zip(example_ids, greedy, answer):
        dial_id = id_.split('/')[1]
        if dial_id != cur_dial_id:
            cur_dial_id = dial_id
            greedy_state = defaultdict()
            answer_state = defaultdict()

        a = a[0]
        a = dataset.span2state(a)
        g = dataset.span2state(g)

        dataset.update_state(a, answer_state)
        dataset.update_state(g, greedy_state)

        full_answer.append(copy.deepcopy(answer_state))
        full_greedy.append(copy.deepcopy(greedy_state))

    return dataset.compute_dst_em(full_greedy, full_answer)


def computeDST_EM(greedy, answer, task):
    # Calculate exact match between diff states
    # Uses dialogues dst metric which takes care of entity normalizations
    dataset_class = getattr(dialogues, task.dataset_name)
    dataset = dataset_class()

    answer = [dataset.span2state(a[0]) for a in answer]
    greedy = [dataset.span2state(g) for g in greedy]
    return dataset.compute_dst_em(greedy, answer)


def compute_metrics(
    predictions: List[str],
    answers: Union[List[str], List[List[str]]],
    requested_metrics: List,
    args,
    example_ids: List[str] = None,
    contexts: List[str] = None,
):
    """
    Inputs:
        predictions: a list of model predictions
        answers: a list of gold answers, each answer can be one item, or a list if multiple gold answers exist
        requested_metrics: contains a subset of the following metrics
            em (exact match)
            # TODO add all
        lang: the language of the predictions and answers. Used for BERTScore.
        args: arguments
        example_ids: used to calculate some of e2e dialogue metrics that need to know span of each dialogue such as JGA
        contexts: used to calculate SER metric that need to know entities in the input
    """
    metric_keys = []
    metric_values = []
    if not isinstance(answers[0], list):
        answers = [[a] for a in answers]
    if 'e2e_dialogue_score' in requested_metrics:
        requested_metrics += [
            a.upper() + '_' + b for (a, b) in zip(args.e2e_dialogue_valid_subtasks, args.e2e_dialogue_valid_submetrics)
        ]
        results = compute_e2e_dialogue_score(predictions, answers, args, example_ids, contexts)
        metric_keys += results.keys()
        metric_values += results.values()
    if 'jga' in requested_metrics:
        jga = computeJGA(predictions, answers, example_ids, args.task)
        metric_keys += ['jga']
        metric_values += [jga]
    if 'ser' in requested_metrics:
        ser = computeSER(predictions, contexts, args.task)
        metric_keys += ['ser']
        metric_values += [ser]
    if 'em' in requested_metrics:
        em = computeEM(predictions, answers)
        metric_keys += ['em']
        metric_values += [em]
    if 'da_em' in requested_metrics:
        da_em = computeDA_EM(predictions, answers, args.task)
        metric_keys += ['da_em']
        metric_values += [da_em]
    if 'dst_em' in requested_metrics:
        dst_em = computeDST_EM(predictions, answers, args.task)
        metric_keys += ['dst_em']
        metric_values += [dst_em]
    if 'casedbleu' in requested_metrics:
        casedbleu = computeCasedBLEU(predictions, answers)
        metric_keys.append('casedbleu')
        metric_values.append(casedbleu)
    if 'bleu' in requested_metrics:
        bleu = computeBLEU(predictions, answers)
        metric_keys.append('bleu')
        metric_values.append(bleu)
    for m in ['rouge1', 'rouge2', 'rougeL']:
        if m in requested_metrics:
            rouge = computeROUGE(predictions, answers, rouge_types=[m])[m]
            requested_metrics.remove(m)
            requested_metrics += [f'{m}_low', f'{m}_mid', f'{m}_high']
            metric_keys += [f'{m}_low', f'{m}_mid', f'{m}_high']
            metric_values += [rouge.low.fmeasure, rouge.mid.fmeasure, rouge.high.fmeasure]

    metric_dict = dict(zip(metric_keys, metric_values))
    metric_dict = OrderedDict((key, metric_dict[key]) for key in requested_metrics)
    return metric_dict


def calculate_metrics(args, validation_output, task):
    metrics_to_compute = task.metrics

    metrics = OrderedDict()
    example_ids = validation_output.example_ids
    predictions = validation_output.predictions
    answers = validation_output.answers
    contexts = validation_output.contexts

    metrics = compute_metrics(
        [p[0] for p in predictions], answers, metrics_to_compute, args, example_ids, contexts
    )  # calculate the metric on all first outputs, all second outputs, etc.

    return metrics
