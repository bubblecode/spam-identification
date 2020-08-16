"""f = w^T * x + b"""
import os

import numpy as np
import numpy.random as random
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def calcWs(alphas, data, labels):
    X = np.mat(data)
    labels = np.mat(labels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labels[i], X[i,:].T)
    return w

# #########################################################
def select_J_rand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j

def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def SMO(data, labels, C, toler, maxIter):
    data_mtx = np.mat(data)
    label_mat = np.mat(labels).transpose()
    b = 0
    m,n = np.shape(data_mtx)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while iter < maxIter:
        alpha_pairs_changed = 0
        print('[{}/{}] begin ...'.format(iter, maxIter))
        for i in range(m):
            fXi = float(np.multiply(alphas, label_mat).T * (data_mtx*data_mtx[i,:].T)) + b
            Ei  = fXi - float(label_mat[i])
            if (label_mat[i]*Ei < -toler and alphas[i] < C) or (label_mat[i]*Ei > toler and alphas[i] > 0):
                j = select_J_rand(i, m)
                fXj = float(np.multiply(alphas,label_mat).T * (data_mtx*data_mtx[j,:].T)) + b
                Ej = fXj - float(label_mat[j])
                alphaI_old = alphas[i].copy()
                alphaJ_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    # print('L == H')
                    continue
                eta = 2.0 * data_mtx[i,:]*data_mtx[j,:].T - data_mtx[i,:]*data_mtx[i,:].T - data_mtx[j,:]*data_mtx[j,:].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= label_mat[j]*(Ei - Ej)/eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJ_old) < 1e-5:
                    # print('j not moving enough')
                    continue
                alphas[i] += label_mat[j]*label_mat[i]*(alphaJ_old - alphas[j])
                b1 = b - Ei - label_mat[i] * \
                     (alphas[i] - alphaI_old)*data_mtx[i,:]*data_mtx[i,:].T - \
                     (alphas[j] - alphaJ_old)*data_mtx[j,:]*data_mtx[j,:].T
                b2 = b - Ej - label_mat[i] * \
                     (alphas[i] - alphaI_old)*data_mtx[i,:]*data_mtx[i,:].T - \
                     (alphas[j] - alphaJ_old)*data_mtx[j,:]*data_mtx[j,:].T
                if alphas[i] > 0 and alphas[i] < C:
                    b = b1
                elif alphas[j] > 0 and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2.0 
                alpha_pairs_changed += 1
                print('iter:{}, i:{}, pairs changed {}'.format(iter,i,alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number: {}'.format(iter))
    return b, alphas

def SVM_samp(train_data, train_label):
    """
    train_data: 转换成词向量后的数据
    train_label: 训练集所对应的标签，（-1表示一般邮件,1表示垃圾邮件）
    return: pred
    """
    b,alphas = SMO(train_data, train_label.T, 0.6, 0.001, 40)
    ws = calcWs(alphas, train_data, train_label.astype(np.int8))
    pred = []
    data_mtx = np.mat(train_data)
    for i in range(len(train_label)):
        classify = data_mtx[i]*np.mat(ws) + b
        pred.append(classify.tolist()[0][0])
    pred = np.array(pred)
    pred[np.where(pred < 0)] == 0
    pred[np.where(pred >=0)] == 1
    return pred

from sklearn import svm
def SVM(train_data,train_label,test_data):
    clf = svm.SVC(gamma='scale')
    clf.fit(train_data, train_label)
    pred = clf.predict(test_data)
    return pred