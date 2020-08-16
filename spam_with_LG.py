import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def sigmoid(data):
    return 1.0/(1+np.exp(-data))

def classify(inputs, weights):
    prob = sigmoid(np.sum(inputs*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def grad_ascent_optim(features, labels):
    features_mtx = np.mat(data=features)
    labels_mtx = np.mat(labels).transpose().astype(np.int8)
    m, n = np.shape(features_mtx)
    alpha = 1e-3
    epochs = 500
    weights = np.ones((n, 1))
    for k in range(epochs):
        h = sigmoid(features_mtx * weights)
        err = (labels_mtx - h)
        weights = weights + alpha * features_mtx.transpose() * err
    return weights

def stoc_grad_ascent_optim(features, labels, epochs=150):
    labels = labels.astype(np.int8)
    m,n = np.shape(features)
    weights = np.ones(n)
    for j in range(epochs):
        data_idx = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_idx = int(np.random.uniform(0, len(data_idx)))
            h = sigmoid(sum(features[rand_idx]*weights))
            err = labels[rand_idx] - h
            weights = weights + alpha * err * features[rand_idx]
            del data_idx[rand_idx]
    return weights


pred = []
def colicTest(train_features, train_labels, test_features, test_labels, e):
    train_weights = stoc_grad_ascent_optim(np.array(train_features), train_labels) # return: (100,)
    errCount = 0
    numTestVec = 0
    global pred
    pred = []
    for i in range(len(test_labels)):
        numTestVec += 1.0
        pred_item = classify(np.array(test_features[i]), train_weights)
        if int(pred_item) != int(test_labels[i]):
            errCount += 1
        pred.append(int(pred_item))
    errRate = (float(errCount)/numTestVec)
    print("Epoch:{}, the error rate of this test is:{}".format(e, errRate))
    return errRate

def LG(train_data, train_label, test_data, test_label):
    """
    return：pred, test_label
    """
    num_test = 10
    err_sum = 0.0
    for k in range(num_test):
        err_sum += colicTest(train_data, train_label, test_data, test_label, k)
    print("average error rate: {}".format(err_sum/float(num_test)))
    return pred, test_label

