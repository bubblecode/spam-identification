import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score


def print_acc_err(pred, label, name):
    acc = len(np.where(pred == label)[0])
    err = len(np.where(pred != label)[0])
    print('[{}] accuracy: {}, error:{}'.format(name, acc/len(label), err/len(label)))

def print_PRF1_score(pred, label, name):
    precision = precision_score(label, pred)
    recall    = recall_score(label, pred)
    f1        = f1_score(label, pred)
    print('[{}] precison:{}, recall:{}, f1_score:{}'.format(name, precision, recall, f1))

def print_PR_F1_score(pred, label, name):
    precision = precision_score(label, pred.round(), average="weighted")
    recall    = recall_score(label, pred.round(), average="weighted")
    f1        = f1_score(label, pred.round(), average="weighted")
    print('[{}] precison:{}, recall:{}, f1_score:{}'.format(name, precision, recall, f1))

def draw_PR(pred, label, name):
    plt.cla()
    precision, recall, thresholds = precision_recall_curve(label, pred)
    plt.plot(recall, precision, 'b')
    plt.title('Precision-Recall Curve - {}'.format(name))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

def draw_ROC(pred, label, name):
    plt.cla()
    fpr, tpr, thresholds = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.title('Receiver operating characteristic - {}'.format(name))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

def draw_confu(pred, labels, name):
    confusion_mtx = confusion_matrix(pred, labels.astype(np.int8).tolist())
    print(confusion_mtx)
    plt.matshow(confusion_mtx)
    plt.title('Confusion matrix - {}'.format(name))
    plt.colorbar()
    plt.ylabel('Label')
    plt.xlabel('Pred')
    plt.show()

def draw_loss_acc(history_dict):
    plt.cla()
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.show()

def split_texts_label(data: np.array):
    """ split texts and labels
    return: texts, labels
    """
    labels = data[:, 0]
    texts  = []
    for i in data:
        texts.append(str(i[1]))
    labels[np.where(labels == 'ham')]  = 0
    labels[np.where(labels == 'spam')] = 1   ## label 0 is normal label 1 is spam
    labels = labels.astype(np.int8)
    return texts, labels

def tfidf_dataset(x_train, x_test):
    ''' tf-idf word embedding
    return: X_train, X_test, tfidf
    '''
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(x_train)  
    X_test = tfidf.transform(x_test)  
    return X_train, X_test, tfidf

def load_data(train_sample: int, val_sample: int, test_sample: int, tokenizer=False):
    """
    train_sample: size of train samples
    val_sample:   size of validation samples
    test_sample:  size of test samples
    tokenizer:    not cut.
    """
    all_train = np.loadtxt('data/SMSSpamCollection.train', delimiter='\t', dtype=str, encoding='utf8')
    all_test  = np.loadtxt('data/SMSSpamCollection.test',  delimiter='\t', dtype=str, encoding='utf8')
    all_val   = np.loadtxt('data/SMSSpamCollection.devel', delimiter='\t', dtype=str, encoding='utf8')
    
    train_data,train_label = split_texts_label(all_train)
    test_data,test_label   = split_texts_label(all_test)
    val_data,val_label     = split_texts_label(all_val)

    if tokenizer:   ## FIXME: https://blog.csdn.net/yyhhlancelot/article/details/86534793
        tk = Tokenizer(num_words=10000)
        tk.fit_on_texts(train_data + val_data + test_data)
        train_data = tk.texts_to_sequences(train_data)
        val_data   = tk.texts_to_sequences(val_data)
        test_data  = tk.texts_to_sequences(test_data)

        word_idx = tk.word_index
        return train_data[:train_sample], train_label[:train_sample], \
               val_data[:val_sample],   val_label[:val_sample], \
               test_data[:test_sample],  test_label[:test_sample], word_idx

    return train_data[:train_sample], train_label[:train_sample], \
           val_data[:val_sample],   val_label[:val_sample], \
           test_data[:test_sample],  test_label[:test_sample]

