import score
import re
import numpy as np
from scipy.sparse import csr_matrix
import time
import matplotlib.pyplot as plt


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


def lossFun(X, y, w, alpha):
    N = X.shape[0]
    y_prediction = sigma(X.dot(w.T)).T
    L = -(1 / N) * np.sum(y * np.log(y_prediction) + (1 - y) * np.log(1 - y_prediction)) + alpha * np.sum(w ** 2)
    return L


def getGradient(X, y, w, alpha):
    N = X.shape[0]
    y_prediction = sigma(X.dot(w.T)).T
    gradient = - (1 / N) * (X.T.dot((y - y_prediction).T)).reshape(w.shape) + 2 * alpha * np.concatenate(([0], w[0][1:]))
    return gradient


def GradientDescent(X, y, w, alpha, epochNum, learningRate, train_texts):
    N = X.shape[0]
    batch_size = 200
    outputFrequency = 500
    LVec = []
    accVec = []
    xVec = []
    batch_amount = N // batch_size
    for i in range(epochNum):
        k = i % batch_amount
        gradient = getGradient(X[k*batch_size:(k+1)*batch_size][:], y[k*batch_size:(k+1)*batch_size], w, alpha)
        w = w + learningRate * gradient
        if (i + 1) % outputFrequency == 0:
            y_prediction = sigma(X.dot(w.T)).T
            L = lossFun(X, y, w, alpha)
            accuracy = 1 - (np.sum(np.abs(np.round(y_prediction) - y)) / N)
            LVec.append(L)
            xVec.append(i+1)
            accVec.append(accuracy)
            print('-'*35)
            print("Epoch: % 3d" % (i+1))
            print("Learning rate: %f" % learningRate)
            print("Loss: %f" % L)
            print("Accuracy: %f" % accuracy)
            print('-'*35)

    # train_texts = np.array(train_texts)
    # print(train_texts[np.round(y_prediction[0][:]) != y])
    return w, LVec, accVec, xVec


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
        previous_word = ''
        for word in doc:
            index = vocab.setdefault(word, len(vocab))
            indices.append(index + 1)
            data.append(1)
            index = vocab.setdefault(previous_word+word, len(vocab))
            indices.append(index + 1)
            data.append(1)
            previous_word = word
        indptr.append(len(indices))
    XMatrix = csr_matrix((data, indices, indptr), dtype=np.uint8)
    n = len(vocab)
    y = np.array(list(map(lambda x: int(x == 'pos'), train_labels)))
    w = weightsinit(n + 1)
    alpha = 2e-4
    learning_rate = 0.15
    w, LVec, accVec, xVec = GradientDescent(XMatrix, y, w, alpha, 5000, learning_rate, train_texts)

    params = ClassificationParameters()
    params.vocab = vocab
    params.w = w
    params.LVec = LVec
    params.xVec = xVec
    params.accVec = accVec

    return params


def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    indptr = [0]
    indices = []
    data = []
    vocab = params.vocab
    w = params.w
    known_words = vocab.keys()
    p = re.compile(r'\W+')
    for i in range(len(texts)):
        indices.append(0)
        data.append(1)
        doc = p.split(texts[i].lower())
        previous_word = ''
        for word in doc:
            if word in known_words:
                index = vocab.setdefault(word, len(vocab))
                indices.append(index + 1)
                data.append(1)
            if previous_word+word in known_words:
                index = vocab.setdefault(previous_word+word, len(vocab))
                indices.append(index + 1)
                data.append(1)
            previous_word = word
        indptr.append(len(indices))
    XMatrix = csr_matrix((data, indices, indptr), dtype=np.uint8, shape=(len(texts), w.shape[1]))
    y_prediction = sigma(XMatrix.dot(w.T)).T
    labels = list(map(lambda x: 'pos' if x == 1 else 'neg', list(np.round(y_prediction[0][:]))))
    return labels


t0 = time.clock()
text = score.load_dataset_fast()
train_ids, train_texts, train_labels = text['train']
test_ids, test_texts, test_labels = text['dev']

params = train(train_texts, train_labels)
LVec = params.LVec
xVec = params.xVec
accVec = params.accVec
print(params.w)
t1 = time.clock()
print('-'*30)
print("Training time: %f" % t1)
print('-'*30)
labels = classify(test_texts, params)

real_labels = np.array(list(map(lambda x: int(x == 'pos'), test_labels)))
predicted_labels = np.array(list(map(lambda x: int(x == 'pos'), labels)))
accuracy = 1 - np.sum(np.abs(real_labels - predicted_labels)) / len(real_labels)
print("Accuracy on test set: %f" % accuracy)

t2 = time.clock()
print('-'*30)
print("Classification time: %f" % (t2 - t1))
print('-'*30)

# plt.plot(xVec, LVec)
# plt.plot(xVec, accVec)
# plt.legend(['Loss', 'Accuracy'])
# plt.show()
