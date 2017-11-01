from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
import mxnet as mx
import random

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def accuracy(output, label):
    predict = output.argmax(axis=1)
    correct = 0.
    iter = 0
    for value in predict:
        if value == 1:
            if label[iter] == value:
                correct+=1
        
        iter+=1
    #1 Precise
    truePrecise = 0.
    if nd.sum(predict) != 0:
        truePrecise = correct / nd.sum(predict).asscalar()
    return nd.mean(predict==label).asscalar(), truePrecise

def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0.
    trueacc =0.
    total = 0
    for data, label in data_iterator:
        output = net(data.as_in_context(ctx))
        tacc, t1acc= accuracy(output, label.as_in_context(ctx))
        acc += tacc
        trueacc += t1acc
        total += 1
    
    if total == 0 :
        return acc
    
    return acc / total, trueacc / total

def load_data_fashion_mnist(batch_size, resize=None):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        if resize:
            # resize to resize x resize
            data = image.imresize(data, resize, resize)
        # change data from height x weight x channel to channel x height x weight
        return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(
        train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.FashionMNIST(
        train=False, transform=transform_mnist)
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(
        mnist_test, batch_size, shuffle=False)
    return (train_data, test_data)

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

def data_iter(X,y, batch_size):
    idx = list(range(len(X)))
    random.shuffle(idx)
    for i in range(0, len(X), batch_size):
        j = nd.array(idx[i:min(i+batch_size,len(X))])
        yield nd.take(X, j), nd.take(y, j)


def trainXY(X_train,y_train, X_test,y_test,x_predict, y_predict, batch_size, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        train_1acc = 0.
        batch = 0
        for data, label in data_iter(X_train,y_train,batch_size):
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                L = loss(output, label)
            L.backward()

            trainer.step(data.shape[0])

            train_loss += nd.mean(L).asscalar()
            ttrain_acc,ttrain_1acc = accuracy(output, label)
            train_acc +=ttrain_acc
            train_1acc+=ttrain_1acc

            batch += 1
            if print_batches and batch % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                    batch, train_loss/batch, train_acc/batch
                ))
                
        test_acc, test_1acc = evaluate_accuracy(data_iter(X_test,y_test,batch_size), net, ctx)
        predict_acc, predict_1acc = evaluate_accuracy(data_iter(x_predict,y_predict,batch_size), net, ctx)
        print("Epoch %d. Loss: %f, Train acc: %f, Test acc: %f, Train True Value acc: %f, Test True Value acc: %f, Predict Acc: %f, Predict True Acc:%f" % (
            epoch, train_loss/batch, train_acc/batch, test_acc,train_1acc / batch, test_1acc,predict_acc,predict_1acc
        ))


def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        batch = 0
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                L = loss(output, label)
            L.backward()

            trainer.step(data.shape[0])

            train_loss += nd.mean(L).asscalar()
            train_acc += accuracy(output, label)

            batch += 1
            if print_batches and batch % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                    batch, train_loss/batch, train_acc/batch
                ))

        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/batch, train_acc/batch, test_acc
        ))
