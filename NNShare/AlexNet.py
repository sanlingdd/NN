# coding=utf-8
from mxnet.gluon import nn
from mxnet import gluon


net = nn.Sequential()
with net.name_scope():
    net.add(
        # firstconv
        nn.Conv2D(channels=96, kernel_size=11,
                  strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),

        #second conv
        nn.Conv2D(channels=256, kernel_size=5,
                  padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),

        #three conv
        nn.Conv2D(channels=384, kernel_size=3,
                  padding=1, activation='relu'),
        nn.Conv2D(channels=384, kernel_size=3,
                  padding=1, activation='relu'),
        nn.Conv2D(channels=256, kernel_size=3,
                  padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),

        #dense
        nn.Flatten(),
        nn.Dense(4096, activation="relu"),
        nn.Dropout(.15),

        nn.Dense(4096, activation="relu"),
        nn.Dropout(.1),
        
        nn.Dense(10)
    )

import sys
sys.path.append('..')
import utils

dataset = gluon.data.ArrayDataset([[[1,1],[2,2],[3,3]],[[1,1],[2,2],[3,3]],[[1,1],[2,2],[3,3]],[[1,1],[2,2],[3,3]],[[1,1],[2,2],[3,3]],[[1,1],[2,2],[3,3]]],[1,0,1,0,1,0])
train_data =gluon.data.DataLoader(dataset, 2, shuffle=True)


train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=96)


from mxnet import init
from mxnet import gluon

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.01})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=5)

