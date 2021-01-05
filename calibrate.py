import xgboost as xgb
import numpy as np
import sklearn
import pickle
from os import path
import torch
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
from matplotlib import pyplot

def pad_features(confidences, f):
    pad_len = max([len(f(c)) for c in confidences])
    all_features = []
    for c in confidences:
        features = f(c)
        # print(features)
        # exit(0)
        all_features.append(np.pad(features, pad_width=(0, pad_len-len(features)), constant_values=0, mode='constant'))

    all_features = np.stack(all_features)
    return all_features

def logit_cv_0(x):
    # return torch.cat([torch.max(x[0].logit_cv).view(-1), torch.max(x[0].logit_variance).view(-1)], dim=0)
    return x[0].logit_cv

def logit_cv_1(x):
    return x[1].logit_cv

def max_var_0(x):
    return x[0].logit_variance.max().view(-1)

def logit_mean_0(x):
    # return torch.cat([torch.max(x[0].logit_cv).view(-1), torch.max(x[0].logit_variance).view(-1)], dim=0)
    return x[0].logit_mean

def nodroplogit_0(x):
    return x[0].nodrop_logits

def logit_mean_1(x):
    return x[1].logit_mean

def logit_var_0(x):
    return x[0].logit_variance

def avg_logprob(x):
    return torch.mean(x[0].nodrop_logits).item()

def tune_and_train(train_dataset, dev_dataset, dev_labels, scale_pos_weight):
    max_depth = [3, 5, 7, 10, 20] # the maximum depth of each tree
    eta = [0.1, 0.5, 0.7] # the training step for each iteration
    num_round = [200]

    best_accuracy = 0
    best_model = None
    best_confusion_matrix = None
    for m, e, n in itertools.product(max_depth, eta, num_round):
        params = {
            'max_depth': m,  
            'eta': e,  
            'objective': 'binary:logistic',
            'eval_metric': 'error',
            'scale_pos_weight': scale_pos_weight
            }
        evals_result = {}
        model = xgb.train(params=params,
                          dtrain=train_dataset,
                          evals=[(dev_dataset, 'dev')],
                          num_boost_round=n, 
                          early_stopping_rounds=50, 
                          evals_result=evals_result,
                          verbose_eval=False)
        # print('best score = ', model.best_score, 'best iteration = ', model.best_iteration)
        # print('evals_result = ', evals_result)
        prediction_probs = extract_confidence_scores(model, dev_dataset)
        predictions = np.round(np.asarray(prediction_probs))
        acc = accuracy_score(dev_labels, predictions)
        confusion_m = confusion_matrix(dev_labels, predictions)
        # print('max_depth = ' + str(m) + ' eta = ' + str(e) + ' num_round = ' + str(n))
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_confusion_matrix = confusion_m
        best_accuracy = max(best_accuracy, acc)

    return best_model, best_accuracy, best_confusion_matrix


def extract_confidence_scores(model, dev_dataset):
    prediction_probs = model.predict(dev_dataset, ntree_limit=model.best_ntree_limit)
    return prediction_probs


# def precision_recall(sorted_confidences, sorted_labels):
#     sorted_labels = np.array(sorted_labels, dtype=np.int)
#     # print(sorted_labels)
#     all_precisions = []
#     all_recalls = []
#     for threshold_idx in range(len(sorted_labels)-1):
#         accepted = sorted_labels[threshold_idx:]
#         rejected = sorted_labels[:threshold_idx]

#         true_positive = np.sum(accepted)
#         false_positive = np.sum(1-accepted)
#         true_negative = np.sum(1-rejected)
#         false_negative = np.sum(rejected)

#         precision = true_positive / (true_positive + false_positive)
#         recall = true_positive / (true_positive + false_negative)
#         all_precisions.append(precision)
#         all_recalls.append(recall)

#     return all_precisions, all_recalls    


def run(confidences, featurizers):
    all_labels = []
    for c in confidences:
        all_labels.append(c[0].first_mistake)
    
    all_features = []
    for featurizer in featurizers:
        padded_feature = pad_features(confidences, featurizer)
        all_features.append(padded_feature)
        # print('padded_feature = ', padded_feature[:,-1].nonzero())
    all_features = np.concatenate(all_features, axis=1)
    # print('all_features = ', all_features.shape)

    all_labels = np.array(all_labels) + 1 # +1 so that minimum is 0
    all_labels = (all_labels == 0)
    # print('all_labels = ', all_labels)
    # avg_logprobs = [avg_logprob(c) for c in confidences]
    all_features_train, all_features_dev, all_labels_train, all_labels_dev = \
        sklearn.model_selection.train_test_split(all_features, all_labels, test_size=0.2, random_state=123)
    dtrain = xgb.DMatrix(data=all_features_train, label=all_labels_train)
    ddev = xgb.DMatrix(data=all_features_dev, label=all_labels_dev)
    # print('ratio of 1s in test set = ', np.sum(all_labels_dev)/len(all_labels_dev))
    scale_pos_weight = np.sum(all_labels_dev)/(np.sum(1-all_labels_dev)) # 1s over 0s
    # print('scale_pos_weight = ', scale_pos_weight)

    best_model, best_accuracy, best_confusion_matrix = tune_and_train(train_dataset=dtrain, dev_dataset=ddev, dev_labels=all_labels_dev, scale_pos_weight=scale_pos_weight)
    print('best dev set accuracy = ', best_accuracy)
    print('best confusion_matrix = ', best_confusion_matrix)

    confidence_scores = extract_confidence_scores(best_model, ddev)

    # sorted_logprobs, sorted_labels = list(zip(*sorted(zip(avg_logprobs_dev, all_labels_dev))))
    # print('sorted_logprobs = ',  sorted_logprobs)
    # print('sorted_labels = ', sorted_labels)
    
    precision, recall, _ = precision_recall_curve(all_labels_dev, confidence_scores)

    return precision, recall


if __name__ == '__main__':
    if path.isfile('confidence.pkl'):
        # load from cache
        with open('confidence.pkl', 'rb') as f:
            confidences = pickle.load(f)
    else:
        exit(1)

    # [([logit_cv_0], 'cv'), ([logit_mean_0], 'mean'), ([logit_var_0], 'variance'), ([nodroplogit_0], 'nodrop logits')]
    for f, name in [([logit_mean_0], 'mean'), ([max_var_0, logit_mean_0], 'mean with var max')]:
        precision, recall = run(confidences, f)
        pyplot.plot(recall, precision, marker='.', label=name)

    avg_logprobs = [avg_logprob(c) for c in confidences]
    all_labels = []
    for c in confidences:
        all_labels.append(c[0].first_mistake)
    all_labels = np.array(all_labels) + 1 # +1 so that minimum is 0
    all_labels = (all_labels == 0)

    all_labels_train, all_labels_dev, avg_logprobs_train, avg_logprobs_dev = \
        sklearn.model_selection.train_test_split(all_labels, avg_logprobs, test_size=0.2, random_state=123)

    logit_precision, logit_recall, _ = precision_recall_curve(all_labels_dev, avg_logprobs_dev)
    pyplot.plot(logit_recall, logit_precision, marker='.', label='average logprob')
    pyplot.legend()
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.savefig('plot.svg')

    
    