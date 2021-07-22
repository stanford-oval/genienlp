from typing import Iterable
from .util import ConfidenceFeatures
from .calibrate import TreeConfidenceEstimator, mean_drop_prob, neg_of, nodrop_prob, nodrop_prob_first_mistake, prob_first_mistake
import numpy as np
import pickle
import math
from tqdm import trange

class PositionEstimator(TreeConfidenceEstimator):
    @staticmethod
    def convert_to_labels(confidences: Iterable[ConfidenceFeatures]):
        labels = []
        for c in confidences:
            labels.append(c[0].first_mistake)
        return labels

    def train_and_validate():
        # no training to be done
        pass

def string_in_grid(array):
    s = ''
    for a in array:
        if isinstance(a, float):
            a = '%.3f' % a
        s += '%15s' % a
    return s

def detect_errors():
    with open(args.confidence_path, 'rb') as f:
        confidences = pickle.load(f)
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    all_predictions = []
    all_scores = []
    for f, name in [
                    ([neg_of(nodrop_prob(0))], 'prob'),
                    ([neg_of(mean_drop_prob(0))], 'mean_prob'),
                    ([prob_first_mistake(0)], 'prob_first_mistake'),
                    ]:
        # logger.info('name = %s', name)
        labels = PositionEstimator.convert_to_labels(confidences)
        position_estimator = PositionEstimator(name=name, featurizers=f, eval_metric=None, mc_dropout_num=0)
        features = position_estimator.convert_to_features(confidences)
        features[features==0] = -np.inf # do not select pad tokens
        predictions = np.argsort(-features, axis=1)[:, 0:1] # take the max
        all_scores.append(-np.sort(-features, axis=1)[:, 0:1])
            
        all_predictions.append((name, predictions))
        count = 0
        correct = 0
        for i in range(len(labels)):
            if labels[i] >= 0:
                count += 1
                for j in range(predictions.shape[1]):
                    if predictions[i][j] == labels[i]:
                        correct += 1
                        break
        print('name = ', name, 'accuracy = %.2f%%' % (correct / count * 100))

    count = 0
    correct = 0
    for i in range(len(confidences)):
        if labels[i] >= 0:
            count += 1
            print('context = ', tokenizer.decode(confidences[i][0].context))
            print(' '*30 + string_in_grid(range(max(len(confidences[i][0].gold_answer), len(confidences[i][0].prediction)))))
            print('mean_drop_prob:              ', string_in_grid(mean_drop_prob(0)(confidences[i]).tolist()))
            print('prob_first_mistake:          ', string_in_grid(prob_first_mistake(0)(confidences[i]).tolist()))
            print('%14s'%'prediction = ', string_in_grid([tokenizer.decode(a) for a in confidences[i][0].prediction]))
            print('%14s'%'gold = ', string_in_grid([tokenizer.decode(a) for a in confidences[i][0].gold_answer]))
            print('%16s'%'first mistake = ', labels[i])
            best_prediction = False
            for idx, (name, predictions) in enumerate(all_predictions):
                print('%20s'%name, 'predicted mistake = ', predictions[i])
                print('%20s'%name, 'scores = ', all_scores[idx][i])
                if predictions[i][0] == labels[i]:
                    best_prediction = True
            if best_prediction:
                correct += 1
            print('-'*100)
    print('best predictor accuracy = ', (correct/count))

def weakest_token(i):
    """
    probability that the weakest token is the first mistake
    """
    def f(x):
        probs = mean_drop_prob(i)(x)
        ret = torch.zeros_like(probs)
        for j in range(len(probs)):
            ret[j] = torch.prod(probs[:j])*(1-probs[j])
        return max(ret)

    return f


import torch
import numpy as np
from .models import TransformerSeq2Seq
from .util import get_devices, load_config_json, set_seed
from genienlp.validate import generate_with_seq2seq_model
from genienlp.tasks.registry import get_tasks
from genienlp.data_utils.example import Example, NumericalizedExamples
from .calibrate import prob_first_mistake, mean_drop_seq_prob, neg_of

def get_error_locations(model, batch: NumericalizedExamples, position_estimator, decoding_error_locations, decoding_top_token, args):
    output = generate_with_seq2seq_model(model, [batch], model.numericalizer, args.tasks, args,
                                         output_predictions_only=True,
                                         output_confidence_features=True,
                                         error=decoding_error_locations,
                                         top_token=decoding_top_token)
    model_predictions = [p[0] for p in output.predictions]
    gold_first_mistakes = [output.confidence_features[i][0].first_mistake for i in range(len(output.confidence_features))]
    error_detection_features = position_estimator.convert_to_features(output.confidence_features)
    error_detection_features[error_detection_features==0] = -np.inf # so that we do not select pad tokens
    sorted_list_of_detected_error = np.argsort(-error_detection_features, axis=1)[:, :] # take the max
    return model_predictions, gold_first_mistakes, sorted_list_of_detected_error

def remove_items_lower_than_threshold(array, thresholds):
    copy_sorted_list_of_detected_error = sorted_list_of_detected_error
    sorted_list_of_detected_error = []
    for i in range(len(array)):
        if len(array[i]) > thresholds[i]:
            sorted_list_of_detected_error.append([a for a in second_sorted_list_of_detected_error[i] if a > copy_sorted_list_of_detected_error[i][mistake_idx]])
        else:
            sorted_list_of_detected_error.append([])

def parse_argv(parser):
    parser.add_argument('--path', type=str, required=True, help='Folder to load the model from')
    parser.add_argument('--input_file', type=str, required=True, help='Input file to read from')
    parser.add_argument('--output_file', type=str, required=True, help='Output file')
    parser.add_argument('--top_tokens', type=int, required=True, help='Number of tokens to consider')
    parser.add_argument('--top_mistakes', type=int, required=True, help='Number of mistakes to consider')
    parser.add_argument('--num_iterations', type=int, default=1,
                        help='Number of iterations to fix mistakes. Each iteration is applied on the output of the previous one.')
    parser.add_argument("--mc_dropout_num", type=int, default=0, help='Number of samples to use for Monte Carlo (MC) dropout. 0 disables MC dropout.')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')

def main(args):
    args.checkpoint_name = 'best.pth'
    args.embeddings = '.embeddings/'
    args.override_confidence_labels = False

    load_config_json(args)
    set_seed(args)
    devices = get_devices()
    device = devices[0] # server only runs on a single device
    model, _ = TransformerSeq2Seq.load(save_directory=args.path,
                                     model_checkpoint_file=args.checkpoint_name,
                                     args=args,
                                     device=device
                                     )

    model.to(device)
    model.eval()

    args.tasks = list(get_tasks(['almond_dialogue_nlu'], args).values())[0]

    all_examples = []
    all_answers = []
    with open(args.input_file) as file:
        for line in file:
            example_id, context, question, answer = tuple([a.strip() for a in line.split('\t')])
            ex = Example.from_raw(str(example_id), context, question, answer, preprocess=args.tasks.preprocess_field, lower=args.lower)
            all_examples.append(ex)
            all_answers.append(answer)
    
    all_features = NumericalizedExamples.from_examples(all_examples, model.numericalizer)
    all_ems = []
    all_predictions = []
    detection_accuracy = [0]*args.top_mistakes
        
    if args.mc_dropout_num > 0:
        name, featurizers = 'prob_first_mistake', [prob_first_mistake(0)]
    else:
        name, featurizers = 'nodrop_prob_first_mistake', [nodrop_prob_first_mistake(0)]
    position_estimator = PositionEstimator(name=name , featurizers=featurizers, eval_metric=None, mc_dropout_num=0) # mc_dropout_num is not used
    with torch.no_grad():
        for batch_idx in trange(math.ceil(len(all_examples)/args.batch_size)):
            batch_features = all_features[batch_idx*args.batch_size : min((batch_idx+1)*args.batch_size, len(all_features))]
            batch_answers = all_answers[batch_idx*args.batch_size : min((batch_idx+1)*args.batch_size, len(all_features))]
            batch_size = len(batch_answers)
            batch = NumericalizedExamples.collate_batches(batch_features, model.numericalizer, device=device)
            batch_previous_error_locations = [-1] * batch_size
            model_predictions, gold_first_mistakes, sorted_list_of_detected_error = get_error_locations(model, batch, position_estimator, None, 1, args=args)
            all_batch_predictions = [[model_predictions[i]] for i in range(batch_size)]
            # print('initial parse = ', prediction)
            for idx in range(args.top_mistakes):
                detection_accuracy[idx] += sum([gold_first_mistakes[i]==sorted_list_of_detected_error[i][idx] for i in range(batch_size)])
            # print('first sorted_list_of_detected_error = ', sorted_list_of_detected_error)
            first_sorted_list_of_detected_error = sorted_list_of_detected_error
            for m_idx in range(args.top_mistakes):
                sorted_list_of_detected_error = first_sorted_list_of_detected_error # reset
                mistake_idx = m_idx
                for token_idx in range(1, args.top_tokens+1):
                    de = [[] for _ in range(batch_size)]
                    for iterations in range(args.num_iterations):
                        if iterations > 0:
                            mistake_idx = 0
                        for i in range(batch_size):
                            if len(sorted_list_of_detected_error[i]) > mistake_idx:
                                de[i].append(sorted_list_of_detected_error[i][mistake_idx]+1)
                        # print('de = ', de)
                        model_predictions, gold_first_mistakes, second_sorted_list_of_detected_error = get_error_locations(model, batch, position_estimator, de, token_idx, args=args)
                        copy_sorted_list_of_detected_error = sorted_list_of_detected_error
                        sorted_list_of_detected_error = []
                        for i in range(len(copy_sorted_list_of_detected_error)):
                            # print('lower bound = ', copy_sorted_list_of_detected_error[i][mistake_idx])
                            if len(copy_sorted_list_of_detected_error[i]) > mistake_idx:
                                sorted_list_of_detected_error.append([a for a in second_sorted_list_of_detected_error[i] if a > copy_sorted_list_of_detected_error[i][mistake_idx]])
                            else:
                                sorted_list_of_detected_error.append([])
                        # print('reparse = ', model_predictions)
                        # print('second sorted_list_of_detected_error = ', sorted_list_of_detected_error)
                        for i in range(batch_size):
                            all_batch_predictions[i].append(model_predictions[i])
            # print('-'*20)
            batch_ems = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                for prediction in all_batch_predictions[i]:
                    batch_ems[i].append(batch_answers[i]==prediction)
            all_ems.extend(batch_ems)
            all_predictions.extend(all_batch_predictions)

    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=1000000)
    # print('all_predictions = ', all_predictions)
    # print('all_features = ', all_features)
    
    all_ems = np.array(all_ems).astype(np.int)
    detection_accuracy = np.array(detection_accuracy).astype(np.float)
    num_examples = all_ems.shape[0]
    print('all_ems = ', all_ems)
    with open(args.output_file, 'w') as output:
        for idx, p in enumerate(all_predictions):
            output.write(all_features[idx].example_id[0] + '\t' + '\t'.join(p) + '\n')

    # remove duplicate correct answers
    for i in range(all_ems.shape[0]):
        seen_correct = False
        for j in range(all_ems.shape[1]):
            if not seen_correct and all_ems[i, j] == 1:
                seen_correct = True
                continue
            if seen_correct:
                all_ems[i, j] = 0
    acc = np.sum(all_ems, axis=0) / num_examples
    print('parse accuracy = ', acc)
    print('top-k parse accuracy = ', np.cumsum(acc))
    print('top-k detection accuracy = ', np.cumsum(detection_accuracy)/num_examples)
    