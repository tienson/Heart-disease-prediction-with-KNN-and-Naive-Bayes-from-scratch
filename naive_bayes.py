import numpy as np
from math import sqrt, pi, log2
import pandas as pd
from collections import Counter

def training(data):
    mean = data.mean(axis = 0)
    std = data.std(axis = 0)
    return mean, std

def continous_likelihood(train, test):
    train = train.to_numpy()
    mean, std = training(train)
    likelihood = 1/ (np.sqrt(2*pi)*std) * np.exp(-np.multiply(test-mean, test-mean)/(2*np.multiply(std,std)))
    return likelihood

def discrete_likelihood(train, test):
    count = {}
    m = len(train)

    for col_name in train.columns:
        count[col_name] = Counter(train[col_name])

    for col_name in test.columns:
        test[col_name] = test[col_name].apply(lambda x: count[col_name][x]/m)

    return test


def naive_bayes(train, test):

    continous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
    continous_train = train[continous_columns]
    discreate_train = train.drop(continous_columns[:-1], axis = 1)

    continous_test = test[continous_columns]
    discreate_test = test.drop(continous_columns[:-1], axis = 1)

    pos_con_train = continous_train[continous_train.target == 1]
    neg_con_train = continous_train[continous_train.target == 0]
    pos_dis_train = discreate_train[discreate_train.target == 1]
    neg_dis_train = discreate_train[discreate_train.target == 0]

    pos_con_likelihood = continous_likelihood(pos_con_train.drop(['target'], axis = 1), continous_test.drop(['target'], axis = 1))
    neg_con_likelihood = continous_likelihood(neg_con_train.drop(['target'], axis = 1), continous_test.drop(['target'], axis = 1))
    pos_dis_likelihood = discrete_likelihood(pos_dis_train.drop(['target'], axis = 1), discreate_test.drop(['target'], axis = 1))
    neg_dis_likelihood = discrete_likelihood(neg_dis_train.drop(['target'], axis = 1), discreate_test.drop(['target'], axis = 1))

    pos_prior = len(train[train.target == 1])/len(train)
    neg_prior = len(train[train.target == 0])/len(train)

    pos_posteriori = np.sum(np.log2(pos_con_likelihood), axis = 1) + np.sum(np.log2(pos_dis_likelihood), axis = 1) + log2(pos_prior)
    neg_posteriori = np.sum(np.log2(neg_con_likelihood), axis = 1) + np.sum(np.log2(neg_dis_likelihood), axis = 1) + log2(neg_prior)

    return np.where(pos_posteriori >= neg_posteriori, 1, 0)

if __name__ == "__main__":
    
    np.seterr(divide = 'ignore') 
    np.random.seed(5)
    data = pd.read_csv("heart.csv")
    (n,m) = data.shape
    data = data.iloc[np.random.permutation(len(data))]
    train = data[:round(0.8*n)]
    test = data[round(0.8*n):]

    predictions = naive_bayes(train, test)
    true_label = test.target.to_numpy()

    TP = TN = FP = FN = right_prediction = 0
    for i in range(0, len(test)):
        right_prediction += 1 if (true_label[i] == predictions[i]) else 0
        TP += 1 if (true_label[i]==1 and predictions[i] == 1) else 0
        TN += 1 if (true_label[i]==0 and predictions[i] == 0) else 0
        FP += 1 if (true_label[i]==0 and predictions[i] == 1) else 0
        FN += 1 if (true_label[i]==1 and predictions[i] == 0) else 0

    accuracy = right_prediction/len(test)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    print("accuracy, precision, recall, F1_score:", round(accuracy, 2), round(precision, 2), round(recall, 2), round(F1_score, 2))