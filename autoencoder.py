import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import chainer
from chainer import datasets, optimizers, training
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset


class Autoencoder(chainer.Chain):
    def __init__(self):
        super().__init__(
            encoder=L.Linear(784, 64),
            decoder=L.Linear(64, 784))

    def __call__(self, x, hidden=False):
        h = F.relu(self.encoder(x))
        if hidden:
            return h
        else:
            return F.relu(self.decoder(h))

def plot(samples):
    for index, (data, label) in enumerate(samples):
        plt.subplot(5, 5, index + 1)
        plt.axis('off')
        plt.imshow(data.reshape(28, 28), cmap=cm.gray_r, interpolation='nearest')
        n = int(label)
        plt.title(n, color='red')
    plt.show()


train, test = datasets.get_mnist()
train = train[0:1000]
train = [i[0] for i in train]
train = tuple_dataset.TupleDataset(train, train)
train_iter = chainer.iterators.SerialIterator(train, 100)
test = test[0:25]

model = L.Classifier(
    Autoencoder(),
    lossfun=F.mean_squared_error
)
model.compute_accuracy = False
optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(
    train_iter, optimizer, device=-1)
trainer = training.Trainer(
    updater,
    (epoch, 'epoch'),
    out='result'
)
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
trainer.extend(extensions.ProgressBar())

trainer.run()

pred_list = []
for (data, label) in test:
    pred_data = model.predictor(np.array([data]).astype(np.float32)).data
    pred_list.append((pred_data, label))
plot(pred_list)
