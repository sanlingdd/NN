# coding=utf-8
import sys
sys.path.append('..')
import utils
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

from mxnet import ndarray as nd

num_inputs = 28*28
num_outputs = 10

num_hidden = 512
weight_scale = .02
W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale)
b2 = nd.zeros(num_hidden)

W3 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3,b3]

for param in params:
    param.attach_grad()

	
def relu(X):
    return nd.maximum(X, 0)

def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = relu(nd.dot(X, W1) + b1)
    h2 = relu(nd.dot(h1, W2) + b2)
    output = nd.dot(h2, W3) + b3
    return output

from mxnet import gluon
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

from mxnet import autograd as autograd

learning_rate = .5

for epoch in range(10):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))

data, label = test_data[0:9]
 # show_images(data)
print('true labels')
print(utils.get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(utils.get_text_labels(predicted_labels.asnumpy()))

