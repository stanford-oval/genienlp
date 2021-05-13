from typing import Iterable
from .util import ConfidenceFeatures
from .calibrate import TreeConfidenceEstimator, mean_drop_prob, neg_of, nodrop_prob, prob_first_mistake
import numpy as np
import pickle

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


def parse_argv(parser):
    parser.add_argument('--path', type=str, required=True, help='Folder to load the model from')
    parser.add_argument('--input_file', type=str, required=True, help='Input file to read from')
    parser.add_argument('--top_tokens', type=int, required=True, help='Number of tokens to consider')
    parser.add_argument('--top_mistakes', type=int, required=True, help='Number of mistakes to consider')
    parser.add_argument("--mc_dropout_num", type=int, default=0, help='Number of samples to use for Monte Carlo (MC) dropout. 0 disables MC dropout.')
    # parser.add_argument('--val_batch_size', nargs='+', default=None, type=int,
                        # help='Batch size for validation corresponding to tasks in val tasks')
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

    task = list(get_tasks(['almond_dialogue_nlu'], args).values())[0]

    with open(args.input_file) as file:
        all_ems = []
        for line in file:
            example_id, context, question, answer = tuple([a.strip() for a in line.split('\t')])
            ex = Example.from_raw(str(example_id), context, question, answer, preprocess=task.preprocess_field, lower=args.lower)

            with torch.no_grad():
                all_features = NumericalizedExamples.from_examples([ex], model.numericalizer)
                batch = NumericalizedExamples.collate_batches(all_features, model.numericalizer, device=device)
                position_estimator = PositionEstimator(name='prob_first_mistake', featurizers=[prob_first_mistake(0)], eval_metric=None, mc_dropout_num=0) # mc_dropout_num is not used
                ems = []
                print('example_id = ', example_id)
                output = generate_with_seq2seq_model(model, [batch], model.numericalizer, task, args, output_predictions_only=True, output_confidence_features=True, error=None)
                prediction =  output.predictions[0][0]
                ems.append(prediction==answer)
                print('initial parse = ', prediction)
                print('correct = ', prediction==answer)
                print('first_mistake = ', output.confidence_features[0][0].first_mistake)
                print('mean_drop_seq_prob = ', mean_drop_seq_prob(0)(output.confidence_features[0]))
                print('weakest_token = ', weakest_token(0)(output.confidence_features[0]))
                confidences = output.confidence_features
                features = position_estimator.convert_to_features(confidences)
                features[features==0] = -np.inf # do not select pad tokens
                detected_error = np.argsort(-features, axis=1)[:, 0:args.top_mistakes] # take the max
                for idx in range(args.top_mistakes):
                    de = detected_error[0][idx]
                    print('detected_error = ', de)
                    print('error prob = ', features[0][de])
                    for token_id in range(1, args.top_tokens+1):
                        print('token_id = ', token_id)
                        output = generate_with_seq2seq_model(model, [batch], model.numericalizer, task, args, output_predictions_only=True, output_confidence_features=True,
                                                    error=[[de+1]], top_token=token_id)
                        prediction =  output.predictions[0][0]
                        print('reparse = ', prediction)
                        print('mean_drop_seq_prob = ', mean_drop_seq_prob(0)(output.confidence_features[0]))
                        print('weakest_token = ', weakest_token(0)(output.confidence_features[0]))

                        print('correct = ', prediction==answer)
                        ems.append(prediction==answer)
                print('-'*20)
                all_ems.append(ems)

        np.set_printoptions(suppress=True)
        np.set_printoptions(threshold=1000000)
        
        all_ems = np.array(all_ems).astype(np.int)
        num_examples = all_ems.shape[0]

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
        print('all_ems = ', all_ems)
    