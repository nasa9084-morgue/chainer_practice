import chainer
import chainer.functions as F
from chainer import links as L
from chainer import report
from chainer import optimizers, iterators
from chainer import datasets, training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from sklearn.datasets import fetch_mldata
import numpy as np

import notmnist

class MLP(chainer.Chain):
    def __init__(self):
        super().__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 10)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class CNN(chainer.Chain):
    def __init__(self, train=True):
        super().__init__(
            # Convolution2D(n_in, n_conv, hw_px, [stride], [padding])
            conv1=L.Convolution2D(1, 32, 5),
            conv2=L.Convolution2D(32, 64, 5),
            l1=L.Linear(1024, 10)
        )

        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        return self.l1(h)


class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super().__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x, t)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

notmnist = notmnist.unpickle('notMNIST_large.pkl')
data = notmnist['data']
target = notmnist['target']
x_train, x_test = np.split(data, [6000])
y_train, y_test = np.split(data, [6000])
train = tuple_dataset.TupleDataset(x_train, y_train)
test = tuple_dataset.TupleDataset(x_test, y_test)
#train, test = datasets.get_mnist(ndim=3)

train_iter = iterators.SerialIterator(
    train,  # dataset
    batch_size=100  # batch size
)
test_iter = iterators.SerialIterator(
    test,  # dataset
    batch_size=100, repeat=False, shuffle=False
)

model = L.Classifier(CNN())
optimizer = optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(
    updater,
    (20, 'epoch'),  # training term (unit=epoch)
    out='result'
)

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
