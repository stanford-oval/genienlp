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

import logging
from collections import Counter, OrderedDict, defaultdict
from typing import Iterable, Union

import sacrebleu
from datasets import load_metric
from dialogues import Bitod
from dialogues.bitod.src.evaluate import convert_lists_to_set
from seqeval import metrics as seq_metrics
from seqeval import scheme as seq_scheme

from .util import requote_program

logger = logging.getLogger(__name__)

# metrics that are calculated over a corpus (i.e. a list of predictions and gold answers, not single ones).
# These metrics cannot be calculated on individual examples and then averaged.
corpus_level_metrics = {'bleu', 'casedbleu', 'ter', 't5_bleu', 'nmt_bleu', 'corpus_f1', 'jga'}


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match(prediction, ground_truth):
    return prediction == ground_truth


def partial_exact_match(prediction, ground_truth):
    prediction = prediction.split()
    ground_truth = ground_truth.split()
    is_correct_token = [p == g for p, g in zip(prediction, ground_truth)]
    partial_score = sum(is_correct_token) / len(is_correct_token)
    return partial_score


def structure_match(prediction, ground_truth):
    return requote_program(prediction) == requote_program(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for idx, ground_truth in enumerate(ground_truths):
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def computeSequenceClassificationPrecision(outputs, targets):
    targets = [target[0] for target in targets]
    precision_metric = load_metric('precision')
    return precision_metric.compute(references=targets, predictions=outputs)['precision']


def computeSequenceClassificationRecall(outputs, targets):
    targets = [target[0] for target in targets]
    recall_metric = load_metric('recall')
    return recall_metric.compute(references=targets, predictions=outputs)['recall']


def computeSequenceClassificationF1(outputs, targets):
    targets = [target[0] for target in targets]
    f1_metric = load_metric('f1')
    return f1_metric.compute(references=targets, predictions=outputs)['f1']


def computeF1(outputs, targets):
    outs = [metric_max_over_ground_truths(f1_score, o, t) for o, t in zip(outputs, targets)]
    return sum(outs) / len(outputs) * 100


def computeEM(outputs, targets):
    outs = [metric_max_over_ground_truths(exact_match, o, t) for o, t in zip(outputs, targets)]
    return sum(outs) / len(outputs) * 100


def computePartialEM(outputs, targets):
    outs = [metric_max_over_ground_truths(partial_exact_match, o, t) for o, t in zip(outputs, targets)]
    return sum(outs) / len(outputs) * 100


def computeSM(outputs, targets):
    outs = [metric_max_over_ground_truths(structure_match, o, t) for o, t in zip(outputs, targets)]
    return sum(outs) / len(outputs) * 100


def computeBERTScore(outputs, targets, lang):
    bertscore_metric = load_metric("bertscore")
    return (
        sum(bertscore_metric.compute(predictions=outputs, references=targets, lang=lang, use_fast_tokenizer=True)['f1'])
        / len(outputs)
        * 100
    )


def computeTER(outputs, targets):
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    ter_metric = sacrebleu.metrics.TER()
    return ter_metric.corpus_score(outputs, targets).score * 100


def computeBLEU(outputs, targets):
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return sacrebleu.corpus_bleu(outputs, targets, lowercase=True).score


def computeCasedBLEU(outputs, targets):
    # lowercase is false
    sacrebleu_metric = load_metric("sacrebleu")
    return sacrebleu_metric.compute(predictions=outputs, references=targets, lowercase=False)['score']


def computeT5BLEU(outputs, targets):
    # tokenize_v14_international is used instead of default tokenize_13a tokenizer
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return sacrebleu.corpus_bleu(
        outputs,
        targets,
        smooth_method="exp",  # default
        smooth_value=0.0,  # default
        force=False,  # default
        lowercase=False,  # default
        tokenize="intl",
        use_effective_order=False,  # default
    ).score


def computeNMTBLEU(outputs, targets):
    # input should be tokenized
    # TODO figure better tokenization esp. for CJK langs

    outputs = [o.split(" ") for o in outputs]
    targets = [[t.split(" ") for t in values] for values in targets]
    bleu_metric = load_metric("bleu")
    return bleu_metric.compute(predictions=outputs, references=targets)['bleu'] * 100


def compute_e2e_dialogue_score(greedy, answer, tgt_lang, args, example_ids):
    num_examples = len(answer)
    subtask_metrics_dict = OrderedDict()

    results = OrderedDict({'e2e_dialogue_score': 0.0, 'JGA': 0.0, 'API_em': 0.0, 'DA_em': 0.0, 'BLEU': 0.0})
    subtask2result_key = OrderedDict({'dst': 'JGA', 'api': 'API_em', 'da': 'DA_em', 'rg': 'BLEU'})

    for k, subtask in enumerate(args.e2e_dialogue_valid_subtasks):
        ids, preds, golds = [], [], []
        for i in range(num_examples):
            id_ = example_ids[i]
            if id_.endswith(f'/{subtask}'):
                ids.append(id_)
                preds.append(greedy[i])
                golds.append(answer[i])

        if golds:
            metrics_to_compute = args.e2e_dialogue_valid_submetrics[k]
            sub_metrics = compute_metrics(preds, golds, [metrics_to_compute], tgt_lang, args, ids)
            subtask_metrics_dict[subtask] = (
                sub_metrics[metrics_to_compute],
                len(golds),
                args.e2e_dialogue_valid_subweights[k],
            )

    # TODO  how should we aggregate?
    weighted_num_examples = 0
    for subtask, (sub_result, num_ex, weight) in subtask_metrics_dict.items():
        result_key = subtask2result_key[subtask]

        results[result_key] += sub_result
        results['e2e_dialogue_score'] += weight * (sub_result * num_ex)
        weighted_num_examples += weight * num_ex

    results['e2e_dialogue_score'] /= weighted_num_examples

    return results


def computeJGA(greedy, answer, example_ids):
    dataset = Bitod()
    hit = 0
    cur_dial_id = None
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

        convert_lists_to_set(answer_state)
        convert_lists_to_set(greedy_state)

        if answer_state == greedy_state:
            hit += 1

    return hit / len(greedy) * 100


def convert_IOB2_to_IOB1(labels):
    cur_category = None
    for n, label in enumerate(labels):
        if label[0] == "B" and label[2:] != cur_category:
            labels[n] = "I" + label[1:]
        cur_category = label[2:]


def compute_ner_f1(predictions, answers, schema='IOB2'):
    predictions_processed = [pred.split() for pred in predictions]
    answers_processed = [ans[0].split() for ans in answers]
    f1 = 0.0

    if schema == 'IOB1':
        convert_IOB2_to_IOB1(predictions_processed)
        convert_IOB2_to_IOB1(answers_processed)
        f1 = (
            seq_metrics.f1_score(y_pred=predictions_processed, y_true=answers_processed, mode='strict', scheme=seq_scheme.IOB1)
            * 100
        )
    elif schema == 'IOB2':
        f1 = seq_metrics.f1_score(y_pred=predictions_processed, y_true=answers_processed) * 100

    return f1


def compute_metrics(
    predictions: Iterable[str],
    answers: Union[Iterable[str], Iterable[Iterable[str]]],
    requested_metrics: Iterable,
    lang: str,
    args,
    example_ids: Iterable[str] = None,
):
    """
    Inputs:
        predictions: a list of model predictions
        answers: a list of gold answers, each answer can be one item, or a list if multiple gold answers exist
        requested_metrics: contains a subset of the following metrics
            em (exact match)
            sm (structure match): valid if the output is ThingTalk code. Whether the gold answer and prediction are identical if we ignore parameter values of ThingTalk programs
            #TODO add all
        lang: the language of the predictions and answers. Used for BERTScore.
        args: arguments
        example_ids: used to calculate some of e2e dialogue metrics that need to know span of each dialogue such as JGA
    """
    metric_keys = []
    metric_values = []
    if not isinstance(answers[0], list):
        answers = [[a] for a in answers]
    if 'e2e_dialogue_score' in requested_metrics:
        requested_metrics += ['JGA', 'API_em', 'DA_em', 'BLEU']
        results = compute_e2e_dialogue_score(predictions, answers, lang, args, example_ids)
        metric_keys += results.keys()
        metric_values += results.values()
    if 'jga' in requested_metrics:
        jga = computeJGA(predictions, answers, example_ids)
        metric_keys += ['jga']
        metric_values += [jga]
    if 'em' in requested_metrics:
        em = computeEM(predictions, answers)
        metric_keys += ['em']
        metric_values += [em]
    if 'pem' in requested_metrics:
        pem = computePartialEM(predictions, answers)
        metric_keys.append('pem')
        metric_values.append(pem)
    if 'sm' in requested_metrics:
        sm = computeSM(predictions, answers)
        metric_keys.append('sm')
        metric_values.append(sm)
    if 'ter' in requested_metrics:
        ter = computeTER(predictions, answers)
        metric_keys.append('ter')
        metric_values.append(ter)
    if 'bertscore' in requested_metrics:
        bertscore = computeBERTScore(predictions, answers, lang)
        metric_keys.append('bertscore')
        metric_values.append(bertscore)
    if 'casedbleu' in requested_metrics:
        casedbleu = computeCasedBLEU(predictions, answers)
        metric_keys.append('casedbleu')
        metric_values.append(casedbleu)
    if 'bleu' in requested_metrics:
        bleu = computeBLEU(predictions, answers)
        metric_keys.append('bleu')
        metric_values.append(bleu)
    if 't5_bleu' in requested_metrics:
        t5_bleu = computeT5BLEU(predictions, answers)
        metric_keys.append('t5_bleu')
        metric_values.append(t5_bleu)
    if 'nmt_bleu' in requested_metrics:
        nmt_bleu = computeNMTBLEU(predictions, answers)
        metric_keys.append('nmt_bleu')
        metric_values.append(nmt_bleu)
    if 'sc_precision' in requested_metrics:
        precision = computeSequenceClassificationPrecision(predictions, answers)
        metric_keys.append('sc_precision')
        metric_values.append(precision)
    if 'sc_recall' in requested_metrics:
        recall = computeSequenceClassificationRecall(predictions, answers)
        metric_keys.append('sc_recall')
        metric_values.append(recall)
    if 'sc_f1' in requested_metrics:
        f1 = computeSequenceClassificationF1(predictions, answers)
        metric_keys.append('sc_f1')
        metric_values.append(f1)
    if 'f1' in requested_metrics:
        f1 = computeF1(predictions, answers)
        metric_keys.append('f1')
        metric_values.append(f1)
    if 'ner_f1_IOB1' in requested_metrics:
        ner_f1_iob1 = compute_ner_f1(predictions, answers, schema='IOB1')
        metric_keys.append('ner_f1_IOB1')
        metric_values.append(ner_f1_iob1)
    if 'ner_f1' in requested_metrics:
        ner_f1 = compute_ner_f1(predictions, answers)
        metric_keys.append('ner_f1')
        metric_values.append(ner_f1)

    metric_dict = dict(zip(metric_keys, metric_values))
    metric_dict = OrderedDict((key, metric_dict[key]) for key in requested_metrics)
    return metric_dict


def calculate_and_reduce_metrics(args, generation_output, metrics_to_compute, lang):
    metrics = OrderedDict()
    example_ids = generation_output.example_ids
    predictions = generation_output.predictions
    answers = generation_output.answers

    if args.reduce_metrics == 'max':
        for i in range(len(predictions[0])):  # for each output (in case of multiple outputs)
            partial_metrics = compute_metrics(
                [p[i] for p in predictions], answers, metrics_to_compute, lang, args, example_ids
            )  # calculate the metric on all first outputs, all second outputs, etc.
            for k, v in partial_metrics.items():
                metrics[k] = max(metrics.get(k, 0), v)
    elif args.reduce_metrics == 'top_k':
        for m in metrics_to_compute:
            if m in corpus_level_metrics:
                logging.warning(
                    f'You are using the corpus-level metric {m} with `--reduce_metrics top_k`, which can lead to incorrect results.',
                )
        for i in range(len(predictions)):  # for each input
            example_metrics = OrderedDict()  # keep track of metrics for one input and all of its outputs
            for j in range(len(predictions[i])):  # for each output (in case of multiple outputs)
                partial_metrics = compute_metrics(
                    [predictions[i][j]], [answers[i]], metrics_to_compute, lang, args, example_ids
                )  # calculate the metric on the j-th output of the i-th input
                for k, v in partial_metrics.items():
                    example_metrics[k] = max(example_metrics.get(k, 0), v)
            # sum metrics for all examples
            for k, v in example_metrics.items():
                metrics[k] = metrics.get(k, 0) + example_metrics[k]
        # convert sums to averages
        for k, v in metrics.items():
            metrics[k] = metrics[k] / len(predictions)
    else:
        raise ValueError('Invalid reduce_metrics argument')

    return metrics
