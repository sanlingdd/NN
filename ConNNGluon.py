# coding=utf-8

from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

import sys
sys.path.append('..')
from utils import load_data_fashion_mnist

batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)


import mxnet as mx

ctx = mx.cpu()


weight_scale = .01

from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Flatten())
    net.add(nn.Dense(128, activation="relu"))
    net.add(nn.Dense(10))

net.initialize()

from mxnet import autograd as autograd
import utils
from mxnet import gluon

learning_rate = .2

from mxnet import autograd
from mxnet import gluon
from mxnet import nd

batch_size = 256

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))