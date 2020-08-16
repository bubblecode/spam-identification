from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


def gaussian_NB(train_data, train_label, test_data):
    '''    acc: 0.21  err：0.79  AUC：0.47
    return: pre_val
    '''
    gnb = GaussianNB()
    gnb.fit(train_data, train_label)
    pre_val = gnb.predict(test_data)
    return pre_val

def complement_NB(train_data, train_label, test_data):
    gnb = ComplementNB()
    gnb.fit(train_data, train_label)
    pre_val = gnb.predict(test_data)
    return pre_val

def bernoulli_NB(train_data, train_label, test_data):
    '''  acc: 0.79  err：0.21  AUC：0.82
    return: pre_val
    '''
    gnb = BernoulliNB()
    gnb.fit(train_data, train_label)
    pre_val = gnb.predict(test_data)
    return pre_val

def multinomial_NB(train_data, train_label, test_data):
    ''' acc: 0.73  err：0.26  AUC：0.66
    return: pre_val
    '''
    gnb = MultinomialNB()
    gnb.fit(train_data, train_label)
    pre_val = gnb.predict(test_data)
    return pre_val
