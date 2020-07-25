from random import random
import collections
import score
import re
import math


def count_labels(labels):
    return {label: sum(1 for l in labels if l == label) for label in set(labels)}


class ClassificationParameters:
    pass


def train(train_texts, train_labels, pretrain_params=None):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """
    vocab_size = 12000
    stop_amount = 20
    alpha = 0.2

    pos_texts = []
    neg_texts = []
    p = re.compile(r'\W+')
    for i in range(len(train_labels)):
        res = p.split(train_texts[i].lower())
        if train_labels[i] == 'pos':
            pos_texts.append(set(res))
        else:
            neg_texts.append(set(res))

    prob_of_pos_class = len(pos_texts) / len(train_labels)
    prob_of_neg_class = 1 - prob_of_pos_class

    pos_counter = collections.Counter()
    for texts in pos_texts:
        for word in list(texts):
            pos_counter[word] += 1

    neg_counter = collections.Counter()
    for texts in neg_texts:
        for word in list(texts):
            neg_counter[word] += 1

    vocab = pos_counter + neg_counter
    vocab = vocab.most_common(vocab_size + stop_amount)
    del vocab[:stop_amount]

    vocab = dict(vocab)
    vocab = collections.Counter(vocab)
    pos_counter = vocab - neg_counter
    neg_counter = vocab - pos_counter
    all_pos = sum(pos_counter.values())
    all_neg = sum(neg_counter.values())

    pos_counter = dict(pos_counter)
    neg_counter = dict(neg_counter)
    vocab = dict(vocab)

    pos_prob = dict()
    neg_prob = dict()
    weight = dict()
    for word in vocab.keys():
        occurs = pos_counter.setdefault(word, 0)
        pos_prob[word] = (alpha + occurs) / (alpha * vocab_size + all_pos)
        occurs = neg_counter.setdefault(word, 0)
        neg_prob[word] = (alpha + occurs) / (alpha * vocab_size + all_neg)
        weight[word] = math.log(pos_prob[word]/neg_prob[word])

    w = collections.Counter(weight)
    print(w.most_common(15))
    print(list(reversed(w.most_common()[-15:-1])))

    res = ClassificationParameters()

    res.pos_prob = pos_prob
    res.neg_prob = neg_prob
    res.prob_of_pos_class = prob_of_pos_class
    res.prob_of_neg_class = prob_of_neg_class
    res.vocab = vocab

    return res


def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    p = re.compile(r'\W+')
    labels = []
    for i in range(len(texts)):
        doc = set(p.split(texts[i].lower()))
        doc = doc.intersection(set(params.vocab.keys()))
        pos_bayes = 0
        neg_bayes = 0
        doc_list = list(doc)
        for word in doc_list:
            pos_bayes += math.log(params.pos_prob[word])
            neg_bayes += math.log(params.neg_prob[word])
        pos_bayes += math.log(params.prob_of_pos_class)
        neg_bayes += math.log(params.prob_of_neg_class)
        if pos_bayes > neg_bayes:
            labels.append('pos')
        else:
            labels.append('neg')

    return labels


text = score.load_dataset_fast()
train_ids, train_texts, train_labels = text['train']
test_ids, test_texts, test_labels = text['dev']

params = train(train_texts, train_labels)


res = classify(test_texts, params)

print(res)
