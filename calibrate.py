import xgboost as xgb
import numpy as np
import sklearn
import pickle
from os import path
import torch
from torch import nn
from torch import optim
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
from matplotlib import pyplot
import argparse

def pad_features(confidences, f):
    pad_len = max([len(f(c)) for c in confidences])
    all_features = []
    all_lengths = []
    for c in confidences:
        features = f(c)
        # print(features)
        all_lengths.append(len(features))
        all_features.append(np.pad(features, pad_width=(0, pad_len-len(features)), constant_values=np.nan, mode='constant'))

    all_features = np.stack(all_features)
    mean = np.nanmean(all_features, axis=0)
    var = np.nanvar(all_features, axis=0)
    # print('all_features = ', all_features)
    all_features = (all_features - mean) / var
    all_features[np.isnan(all_features)] = 0
    # print('all_features = ', all_features)
    # exit(0)
    return all_features, all_lengths

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

def length_0(x):
    return torch.tensor(len(x[0].logit_mean)).view(-1)

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
            'eval_metric': 'aucpr',
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
        print('best dev score = ', model.best_score, 'best iteration = ', model.best_iteration)
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


def run(confidences, featurizers):
    all_labels = []
    for c in confidences:
        all_labels.append(c[0].first_mistake)
    
    all_features = []
    for featurizer in featurizers:
        padded_feature, _ = pad_features(confidences, featurizer)
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
    pass_rate, accuracies = accuracy_at_pass_rate(all_labels_dev, confidence_scores)

    return precision, recall, pass_rate, accuracies


def accuracy_at_pass_rate(labels, confidence_scores):
    sorted_confidence_scores, sorted_labels = zip(*sorted(zip(confidence_scores, labels)))
    sorted_labels = np.array(sorted_labels, dtype=np.int)
    # print('sorted_confidence_scores = ', sorted_confidence_scores)
    # print('sorted_labels = ', sorted_labels)
    all_pass_rates = []
    all_accuracies = []
    for i in range(len(sorted_labels)):
        pass_labels = sorted_labels[i:]
        pass_rate = len(pass_labels) / len(sorted_labels)
        all_pass_rates.append(pass_rate)
        accuracy = np.sum(pass_labels) / len(pass_labels)
        all_accuracies.append(accuracy)

    return all_pass_rates, all_accuracies


class LSTMEstimator(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)


    def forward(self, x, x_lengths):
        # print('x.shape = ', x.shape)
        # print('x_lengths = ', x_lengths)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, self.hidden_cell = self.lstm(x)
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # print('lstm_out.shape = ', lstm_out.shape)
        lengths = lengths.unsqueeze(-1).unsqueeze(-1)
        # print('lengths.shape = ', lengths.shape)
        lengths = lengths.expand(lstm_out.shape[0], lstm_out.shape[1], lstm_out.shape[2]).to(lstm_out.device)
        # print('lengths.shape = ', lengths.shape)
        # print('lengths = ', lengths)
        lstm_out = lstm_out.contiguous()
        # print('lstm_out = ', lstm_out)
        lstm_out = lstm_out.gather(1, lengths-1)[:,0,:]
        # print('after gather: lstm_out.shape = ', lstm_out.shape)
        # print('lstm_out = ', lstm_out)
        predictions = self.linear(lstm_out)
        return predictions


class ConfidenceDataset(torch.utils.data.Dataset):

    def __init__(self, features: torch.Tensor, labels: torch.Tensor, feature_lengths: torch.Tensor):
        assert features.shape[0] == labels.shape[0], 'Batch size differes in features and labels. features.shape = ' + \
                                                      str(features.shape) + ' labels.shape = ' + str(labels.shape[0])
        self.features = features
        self.labels = labels
        self.feature_lengths = feature_lengths
        # print('features = ', self.features)
        # print('labels = ', self.labels)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # print('features = ', self.features[idx, :])
        # print('labels = ', self.labels[idx, :])
        # print('feature_lengths = ', self.feature_lengths[idx])
        return self.features[idx, :], self.labels[idx, :], self.feature_lengths[idx]

    # def collate_fn(self, batch):
    #     pass
        

def train_LSTM(train_features: np.array, train_labels: np.array, dev_features: np.array, dev_labels: np.array, train_feature_lengths, dev_feature_lengths):
    train_features = train_features[0:]
    train_labels = train_labels[0:]
    dev_features = dev_features[0:]
    dev_labels = dev_labels[0:]
    train_feature_lengths = train_feature_lengths[0:]
    dev_feature_lengths = dev_feature_lengths[0:]

    device = torch.device('cuda:0')
    train_features = torch.tensor(train_features, device=device)
    train_labels = torch.tensor(train_labels, device=device).view(-1, 1).float()
    dev_features = torch.tensor(dev_features, device=device)
    dev_labels = torch.tensor(dev_labels, device=device).view(-1, 1).float()
    train_feature_lengths = torch.tensor(train_feature_lengths)
    dev_feature_lengths = torch.tensor(dev_feature_lengths)

    # print('dev_feature_lengths = ', dev_feature_lengths)

    if len(train_features.shape) == 2:
        train_features = train_features.unsqueeze(-1) # feature_size is 1, so add another dimension
        dev_features = dev_features.unsqueeze(-1) # feature_size is 1, so add another dimension

    model = LSTMEstimator(hidden_layer_size=100)
    model.to(device)
    train_set = ConfidenceDataset(train_features, train_labels, train_feature_lengths)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    dev_set = ConfidenceDataset(dev_features, dev_labels, dev_feature_lengths)
    dev_data_loader = torch.utils.data.DataLoader(dev_set, batch_size=100, shuffle=False)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    max_accuracy = 0
    for epoch in range(200):
        model.train()
        for features, labels, feature_lengths in train_data_loader:
            model.zero_grad()
            confidence_scores = model(features, feature_lengths)
            loss = loss_function(confidence_scores, labels)
            # print('confidence_scores = ', confidence_scores)
            print('loss = ', loss.item())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            accuracy = 0
            for features, labels, feature_lengths in dev_data_loader:
                confidence_scores = model(features, feature_lengths)
                predictions = torch.sigmoid(confidence_scores).round()
                
                # print('confidence_scores = ', confidence_scores)
                # print('labels = ', labels)
                # print('predictions = ', predictions)
                accuracy += torch.sum(predictions==labels)
                # print('confidence_score = ', confidence_scores)
            accuracy = accuracy/len(dev_data_loader)
            max_accuracy = max(max_accuracy, accuracy)
            print('accuracy = ', accuracy)
    print('max_accuracy = ', max_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidence_path', type=str, help='The path to the pickle file where the list of ConfidenceOutput objects is saved')
    # parser.add_argument('--save', type=str, help='Where to save the calibrator model after training')
    args = parser.parse_args()

    if path.isfile(args.confidence_path):
        # load from cache
        with open(args.confidence_path, 'rb') as f:
            confidences = pickle.load(f)
    else:
        exit(1)



    all_labels = []
    for c in confidences:
        all_labels.append(c[0].first_mistake)
    
    all_features = []
    for featurizer in [logit_mean_0]:
        padded_feature, feature_lengths = pad_features(confidences, featurizer)
        all_features.append(padded_feature)
        # print('padded_feature = ', padded_feature[:,-1].nonzero())

    all_features = np.concatenate(all_features, axis=1)
    # print('normalized all_features = ', all_features)

    all_labels = np.array(all_labels) + 1 # +1 so that minimum is 0
    all_labels = (all_labels == 0)
    # print('all_labels = ', all_labels)
    # avg_logprobs = [avg_logprob(c) for c in confidences]
    all_features_train, all_features_dev, all_labels_train, all_labels_dev, feature_lengths_train, feature_lengths_dev = \
        sklearn.model_selection.train_test_split(all_features, all_labels, feature_lengths, test_size=0.2, random_state=123)

    train_LSTM(all_features_train, all_labels_train, all_features_dev, all_labels_dev, feature_lengths_train, feature_lengths_dev)




    # # [([logit_cv_0], 'cv'), ([logit_mean_0], 'mean'), ([logit_var_0], 'variance'), ([nodroplogit_0], 'nodrop logits')]
    # for f, name in [([logit_mean_0], 'mean'), ([length_0, logit_mean_0], 'mean, length')]:
    #     precision, recall, pass_rate, accuracies = run(confidences, f)
    #     pyplot.figure(0)
    #     pyplot.plot(recall, precision, marker='.', label=name)
    #     pyplot.figure(1)
    #     pyplot.plot(pass_rate, accuracies, marker='.', label=name)
        

    # avg_logprobs = [avg_logprob(c) for c in confidences]
    # all_labels = []
    # for c in confidences:
    #     all_labels.append(c[0].first_mistake)
    # all_labels = np.array(all_labels) + 1 # +1 so that minimum is 0
    # all_labels = (all_labels == 0)

    # all_labels_train, all_labels_dev, avg_logprobs_train, avg_logprobs_dev = \
    #     sklearn.model_selection.train_test_split(all_labels, avg_logprobs, test_size=0.2, random_state=123)

    # logit_precision, logit_recall, _ = precision_recall_curve(all_labels_dev, avg_logprobs_dev)
    # pyplot.figure(0)
    # pyplot.plot(logit_recall, logit_precision, marker='.', label='average logprob')
    # pyplot.legend()
    # pyplot.grid()
    # pyplot.xlabel('Recall')
    # pyplot.ylabel('Precision')
    # pyplot.savefig('precision-recall.png')

    # pass_rates, accuracies = accuracy_at_pass_rate(all_labels_dev, avg_logprobs_dev)
    # pyplot.figure(1)
    # pyplot.plot(pass_rates, accuracies, marker='.', label='average logprob')
    # pyplot.legend()
    # pyplot.grid()
    # pyplot.xlabel('Pass Rate')
    # pyplot.ylabel('Accuracy')
    # pyplot.savefig('pass-acc.png')
    