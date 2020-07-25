import score
from classifier import train
import pickle
import matplotlib.pyplot as plt


text = score.load_dataset_fast()
train_ids, train_texts, train_labels = text['train']
test_ids, test_texts, test_labels = text['dev']

lrates = [0.001]
layer_sizes = [[200, 100, 20, 10]]
epoch = 601
legend = []

xVec = range(0, epoch, 200)

for size in layer_sizes:
    for rate in lrates:
        params, ceVec = train(train_texts[:], train_labels[:], epochs=epoch, learning_rate=rate, layers_size=size)
        with open("W" + str(size) + str(round(rate * 1000)) + ".pickle", 'wb') as f:
            pickle.dump(params, f)
        plt.plot(xVec, ceVec)
        legend.append(str(rate) + ' ' + str(size))

plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Rate curve')
plt.legend(legend)
plt.show()
