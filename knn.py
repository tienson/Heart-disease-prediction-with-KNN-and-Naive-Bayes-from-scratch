import numpy as np
from math import sqrt
import pandas as pd

def preprocess_data():

    

    data = pd.read_csv("heart.csv")
    target = data.target
    data = data.drop('target', axis = 1)

    continous_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] 
    
    data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'], prefix = ['cp', 'restecg', 'slope', 'thal'])
    
    data[continous_columns] = (data[continous_columns] - data[continous_columns].mean(axis=0)) / data[continous_columns].std(axis=0)
    data = pd.concat([data, target], axis=1, sort=False)

    # print(data)
    return data

def distance(r1, r2):
	distance = 0.0
	for i in range(len(r1)-2):
		distance += (r1[i] - r2[i])**2
	return sqrt(distance)

def KNN(k, validation, train):
    TP = TN = FP = FN = right_prediction = 0
    for val_row in validation:
            distances = []
            for train_row in train:
                distances.append([distance(val_row, train_row), train_row[-1]])
            distances.sort()
            predict_class = round(sum(np.array(distances[:k]))[1] / k)
            right_prediction += 1 if (val_row[-1] == predict_class) else 0
            TP += 1 if (val_row[-1]==1 and predict_class == 1) else 0
            TN += 1 if (val_row[-1]==0 and predict_class == 0) else 0
            FP += 1 if (val_row[-1]==0 and predict_class == 1) else 0
            FN += 1 if (val_row[-1]==1 and predict_class == 0) else 0

    accuracy = right_prediction/len(validation)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, F1_score


if __name__ == "__main__":
    np.random.seed(1)
    data = preprocess_data()
    (n,m) = data.shape
    data = data.iloc[np.random.permutation(len(data))]
    train = data[:round(0.6*n)].to_numpy()
    validation  = data[round(0.6*n):round(0.8*n)].to_numpy()
    test = data[round(0.8*n):].to_numpy()

    num_val = n*0.2
    best_accuracy = 0
    # using train and validation set to get best k
    for k in range(1,10):
        accuracy, precision, recall, F1_score = KNN(k, validation, train)
        if (best_accuracy < accuracy):
            best_accuracy = accuracy
            best_k = k
    # use k to predict on test set
    accuracy, precision, recall, F1_score = KNN(best_k, test, train)
    print("best number of neighbors = ", best_k)
    print("accuracy, precision, recall, F1_score:", round(accuracy, 2), round(precision, 2), round(recall, 2), round(F1_score, 2))


        