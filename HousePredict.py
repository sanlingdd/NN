import pandas as pd
import numpy as np
import math

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
all_X = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))


numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean())
                                                            / (x.std()))

handledTrain = all_X[:]

all_X = pd.get_dummies(all_X, dummy_na=True)


all_X = all_X.fillna(all_X.mean())

num_train = train.shape[0]

X_train = all_X[:num_train].as_matrix()
X_test = all_X[num_train:].as_matrix()
y_train = train.SalePrice.as_matrix()

handledTrain = all_X[:num_train]

handledTrain = all_X[num_train:]


from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

X_train = nd.array(X_train)
y_train = nd.array(y_train)
y_train.reshape((num_train, 1))

X_test = nd.array(X_test)

square_loss = gluon.loss.L2Loss()

def get_rmse_log(net, X_train, y_train):
    num_train = X_train.shape[0]
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * nd.sum(square_loss(
        nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)


def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(32,activation="relu"))
        net.add(gluon.nn.Dense(16,activation="relu"))
 #       net.add(gluon.nn.Dense(32,activation="relu"))
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

def train(net, X_train, y_train, X_test, y_test, epochs,
          verbose_epoch, learning_rate, weight_decay,batch_size,stopSteps):
    train_loss = []
    if X_test is not None:
        test_loss = []

    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size,shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate,
                             'wd': weight_decay})
    net.collect_params().initialize(force_reinit=True)
    miniumTestLoss = 1000
    iter = 0
    currentIter = 0
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)



        cur_train_loss = get_rmse_log(net, X_train, y_train)
        cur_train_loss = get_rmse_log(net, X_test, y_test)
        ##early stop
        iter +=1
        if(cur_train_loss < miniumTestLoss):
            miniumTestLoss = cur_train_loss
            currentIter = iter
        elif(iter - currentIter > stopSteps):
            break

        if epoch > verbose_epoch:
            print("Epoch %d, train loss: %f, test loss: %f" % (epoch, cur_train_loss, cur_test_loss))
        train_loss.append(cur_train_loss)
        if X_test is not None:
            cur_test_loss = get_rmse_log(net, X_test, y_test)
            test_loss.append(cur_test_loss)

    plt.plot(train_loss)
    plt.legend(['train'])
    if X_test is not None:
        plt.plot(test_loss)
        plt.legend(['train','test'])
    plt.show()
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss


data_train_size = int(num_train // 4)

X_val_test = X_train[0: data_train_size, :]
y_val_test = y_train[0: data_train_size]

X_val_train = X_train[data_train_size:num_train, :]
y_val_train = y_train[data_train_size:num_train]



epochs = 2500
verbose_epoch = 1
learning_rate = 0.0065
weight_decay = 0.0295
batch_size = 250
stopSteps = 20

def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
          weight_decay):
    net = get_net()
    train(net, X_val_train, y_val_train, X_val_test, y_val_test, epochs, verbose_epoch,
          learning_rate, weight_decay,batch_size,stopSteps)
    preds = net(X_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
          weight_decay)

