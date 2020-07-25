import numpy as np
import score
from classifier import classify
import pickle

text = score.load_dataset_fast()
train_ids, train_texts, train_labels = text['train']
test_ids, test_texts, test_labels = text['dev']

with open('W[200, 100, 20, 10]30.pickle', 'rb') as f:
    params = pickle.load(f)

labels = classify(train_texts[:], params)

real_labels = np.array(list(map(lambda x: int(x == 'pos'), train_labels[:])))
predicted_labels = np.array(list(map(lambda x: int(x == 'pos'), labels)))
accuracy = 1 - np.sum(np.abs(real_labels - predicted_labels)) / len(real_labels)
print(accuracy)

labels = classify(test_texts[:], params)

real_labels = np.array(list(map(lambda x: int(x == 'pos'), test_labels[:])))
predicted_labels = np.array(list(map(lambda x: int(x == 'pos'), labels)))
accuracy = 1 - np.sum(np.abs(real_labels - predicted_labels)) / len(real_labels)
print(accuracy)
