# coding=utf-8
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)


#for data, label in data_iter:
#    print(data, label)

## NN sequential
net = gluon.nn.Sequential()

## One layer NN
net.add(gluon.nn.Dense(1))

##Random initialization
net.initialize()

##Loss function
square_loss = gluon.loss.L2Loss()

## define train model, train with sgd, store parameters
trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.1})

## T
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))

#NN after trained
dense = net[0]
## parameters
print(true_w,"\n",dense.weight.data())

##Bias
print(true_b,"\n", dense.bias.data())


