from random import random
import collections
import score
import re
import math
import numpy as np
from scipy.sparse import csr_matrix
import time

def count_labels(labels):
    return {label: sum(1 for l in labels if l == label) for label in set(labels)}


class ClassificationParameters:
    pass


def sigma(z):
    eps = 1e-10
    return (1 - eps)/(1 + np.exp(z)) + eps/2


def weightsinit(n):
    w = np.array(np.zeros((1, n)))
    return w


def lossfun(X, y, w, alpha):
    N = X.shape[0]
    y_prediction = sigma(X.dot(w.T)).T
    L = -(1/N)*np.sum(y * np.log(y_prediction) + (1 - y)*np.log(1 - y_prediction)) + alpha*np.sum(w**2)
    accuracy = 1 - (np.sum(np.abs(np.round(y_prediction) - y)) / N)
    gradient = - (1 / N) * (X.T.dot((y - y_prediction).T)).reshape(w.shape) + 2 * alpha * np.concatenate(([0], w[0][1:]))
    return L, accuracy, gradient


def GradientDescent(X, y, w, alpha, epochNum, learningRate):
    for i in range(epochNum):
        L, accuracy, gradient = lossfun(X, y, w, alpha)
        w = w + learningRate * gradient
        if (i + 1) % 500 == 0:
            print("Epoch: % 3d" % (i+1))
            print("Learning rate: %f" % learningRate)
            print("Loss: %f" % L)
            print("Accuracy: %f" % accuracy)
            if accuracy == 1.:
                print("********Horay!!! Absolute accuracy, mthfk3r!!!*******")
                break
    return w


def train(train_texts, train_labels, pretrain_params=None):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    indptr = [0]
    indices = []
    data = []
    vocab = {}
    p = re.compile(r'\W+')
    for i in range(len(train_labels)):
        indices.append(0)
        data.append(1)
        doc = p.split(train_texts[i].lower())
        for word in doc:
            index = vocab.setdefault(word, len(vocab))
            indices.append(index + 1)
            data.append(1)
        indptr.append(len(indices))
    XMatrix = csr_matrix((data, indices, indptr), dtype=np.uint8)
    n = len(vocab)
    y = np.array(list(map(lambda x: int(x == 'pos'), train_labels)))
    w = weightsinit(n + 1)
    alpha = 0
    w = GradientDescent(XMatrix[:1000][:], y[:1000], w, alpha, 10000, 0.01)

    return w


def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    p = re.compile(r'\W+')
    labels = []
    vocab = set(params.vocab.keys())
    for i in range(len(texts)):
        doc = p.split(texts[i].lower())
        N = len(doc)
        pos_bayes = math.log(math.factorial(N))
        neg_bayes = pos_bayes
        doc_set = set(doc)
        for word in doc_set:
            k = doc.count(word)
            if word not in vocab:
                p_pos = alpha / (alpha * (vocab_size + 1) + params.all_pos)
                p_neg = alpha / (alpha * (vocab_size + 1) + params.all_neg)
            else:
                p_pos = params.pos_prob[word]
                p_neg = params.neg_prob[word]

            k_log_fact = math.log(math.factorial(k))
            pos_bayes += k*math.log(p_pos) - k_log_fact
            neg_bayes += k*math.log(p_neg) - k_log_fact
        pos_bayes += math.log(params.prob_of_pos_class)
        neg_bayes += math.log(params.prob_of_neg_class)
        if pos_bayes > neg_bayes:
            labels.append('pos')
        else:
            labels.append('neg')

    return labels


t0 = time.clock()
text = score.load_dataset_fast()
train_ids, train_texts, train_labels = text['train']
test_ids, test_texts, test_labels = text['dev']

w = train(train_texts, train_labels)
print(w)
print("Training time: %f" % time.clock())
