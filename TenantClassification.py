import sys
sys.path.append('..')

from mxnet import gluon
from mxnet import ndarray as nd
import pandas as pd
import numpy as np
import random


data = pd.read_excel("data/All1605.xlsx", header=None,
                     names=['Purpose', 'UserCount', 'Order', 'Delivery', 'Quotation', 'Purchase', 'Opptunity', 'Customer','Contact','AllProduct','ActiveProduct','InactiveProduct','AllSKU','ActiveSKU','InactiveSKU','Status'],
                     dtype={'Purpose':object, 'UserCount':int, 'Order':int, 'Delivery':int, 'Quotation':int, 'Purchase':int, 'Opptunity':int,'Customer':int,'Contact':int,'AllProduct':int,'ActiveProduct':int,'InactiveProduct':int,'AllSKU':int,'ActiveSKU':int,'InactiveSKU':int,'Status':int
                            }
                   )

all_X = data.loc[:, 'Purpose':'InactiveSKU']
numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean())
                                                            / (x.std()))
all_X = pd.get_dummies(all_X, dummy_na=True)

all_X = all_X.fillna(all_X.mean())

#all_X['Status'] = data['Status']
idx = list(all_X.index)
random.shuffle(idx)
all_X = all_X.iloc[idx]

num_train = len(all_X) // 4 * 3
num_test = len(all_X) - num_train;

X_train = nd.array(all_X[:num_train].as_matrix())
y_train = nd.array(data.Status[:num_train].as_matrix())

X_test = nd.array(all_X[num_train:].as_matrix())
y_test = nd.array(data.Status[num_train:].as_matrix())

from mxnet.gluon import nn
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        #nn.Dropout(0.5),
        nn.Dense(128, activation="relu"),
        #nn.Dropout(0.5),
        nn.Dense(2)
    )
import utils

from mxnet import init
from mxnet import gluon

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.001})

batch_size = 16
utils.trainXY(X_train,y_train, X_test,y_test,batch_size, net, loss,trainer, ctx, num_epochs=10)

