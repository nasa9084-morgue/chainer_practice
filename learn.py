import chainer
import chainer.functions as F
from chainer import links as L
from chainer import report
from chainer import optimizers, iterators
from chainer import datasets, training
from chainer.training import extensions
from sklearn.datasets import fetch_mldata
import numpy as np


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


train, test = datasets.get_mnist(ndim=3)

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
