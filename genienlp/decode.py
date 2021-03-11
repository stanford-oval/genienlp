import torch
import numpy as np
from .models import TransformerSeq2Seq
from .util import init_devices, load_config_json, set_seed
from genienlp.validate import generate_with_model
from genienlp.tasks.registry import get_tasks
from genienlp.data_utils.example import Example, NumericalizedExamples
from genienlp.calibrate import ConfidenceEstimator, PositionEstimator, prob_first_mistake, mean_drop_seq_prob, nodrop_seq_prob, weakest_token, neg_of


def parse_argv(parser):
    parser.add_argument('--path', type=str, required=True, help='Folder to load the model from')
    parser.add_argument('--input_file', type=str, required=True, help='Input file to read from')
    parser.add_argument('--top_tokens', type=int, required=True, help='Number of tokens to consider')
    parser.add_argument('--top_mistakes', type=int, required=True, help='Number of mistakes to consider')
    parser.add_argument("--mc_dropout_num", type=int, default=0, help='Number of samples to use for Monte Carlo (MC) dropout. 0 disables MC dropout.')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')

def main(args):
    args.checkpoint_name = 'best.pth'
    args.embeddings = '.embeddings/'
    args.override_confidence_labels = False
    estimator = ConfidenceEstimator.load('./calibrator.calib')
    criterion = lambda x: -estimator.estimate(x)[0]

    load_config_json(args)
    set_seed(args)
    devices = init_devices(args)
    device = devices[0] # server only runs on a single device
    model, _ = TransformerSeq2Seq.from_pretrained(args.path,
                                     model_checkpoint_file=args.checkpoint_name,
                                     args=args,
                                     device=device
                                     )

    model.to(device)
    model.eval()

    task = list(get_tasks(['almond_dialogue_nlu'], args).values())[0]

    with open(args.input_file) as file:
        all_ems = []
        all_cs = []
        for line in file:
            example_id, context, question, answer = tuple([a.strip() for a in line.split('\t')])
            ex = Example.from_raw(str(example_id), context, question, answer, preprocess=task.preprocess_field, lower=args.lower)

            with torch.no_grad():
                all_features = NumericalizedExamples.from_examples([ex], model.numericalizer, args.add_types_to_text)
                batch = NumericalizedExamples.collate_batches(all_features, model.numericalizer, device=device, db_unk_id=args.db_unk_id)
                position_estimator = PositionEstimator(name='prob_first_mistake', featurizers=[prob_first_mistake(0)], eval_metric=None, mc_dropout_num=0)
                ems = []
                cs = []
                print('example_id = ', example_id)
                output = generate_with_model(model, [batch], model.numericalizer, task, args, output_predictions_only=True, output_confidence_features=True, error=None)
                prediction =  output.predictions[0][0]
                ems.append(prediction==answer)
                cs.append(criterion(output.confidence_features))
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
                        output = generate_with_model(model, [batch], model.numericalizer, task, args, output_predictions_only=True, output_confidence_features=True,
                                                    error=[[de+1]], top_token=token_id)
                        prediction =  output.predictions[0][0]
                        print('reparse = ', prediction)
                        print('mean_drop_seq_prob = ', mean_drop_seq_prob(0)(output.confidence_features[0]))
                        print('weakest_token = ', weakest_token(0)(output.confidence_features[0]))

                        print('correct = ', prediction==answer)
                        ems.append(prediction==answer)
                        cs.append(criterion(output.confidence_features))
                print('-'*20)
                all_ems.append(ems)
                all_cs.append(cs)

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
        best_idx = np.argsort(all_cs, axis=1)[:, 0]
        all_cs = np.array(all_cs)
        acc = np.sum(all_ems, axis=0) / num_examples
        print('parse accuracy = ', acc)
        print('top-k parse accuracy = ', np.cumsum(acc))
        print('all_ems = ', all_ems)
        print('all_cs = ', all_cs)
        print('best_idx = ', best_idx)
        best_ems = np.array([all_ems[i, best_idx[i]] for i in range(num_examples)])
        print('best_ems = ', best_ems)
        print('best_ems accuracy = ', np.sum(best_ems)/num_examples)