from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
import mxnet as mx
import random
from datetime import datetime
from mxnet import ndarray as nd


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def accuracy(output, label, threshold=0.95):
    predict = output.argmax(axis=1)
    confidences = nd.softmax(output).max(axis=1)
    
    correct = 0.
    predictOne = 0.
    for value, confidence,labelelement in zip(predict,confidences,label):
        if value == 1 and confidence >= threshold:
            predictOne += 1
            if labelelement == value:
                correct += 1
            
    #1 Precise
    truePrecise = 0.
    if predictOne != 0.:
        truePrecise = correct / predictOne
    
    return nd.mean(predict==label).asscalar(), truePrecise, predictOne != 0. 

def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0.
    trueacc =0.
    total = 0
    trueTotal=0
    for data, label in data_iterator:
        output = net(data.as_in_context(ctx))
        tacc, t1acc, Predicted= accuracy(output, label.as_in_context(ctx))
        acc += tacc
        trueacc += t1acc
        total += 1
        if Predicted:
            trueTotal+=1
    
    if total == 0:
        return 0., 0.
    if trueTotal ==0:
        return acc / total, 0.

    return acc / total, trueacc / trueTotal

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

def trainXY(X_train,y_train, X_test,y_test,x_predict, y_predict, batch_size, net, loss, trainer, ctx, num_epochs,name,print_batches=None):
    """Train a network"""
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        train_1acc = 0.
        batch = 0
        trueBatch = 0
        for data, label in data_iter(X_train,y_train,batch_size):
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                L = loss(output, label)
            L.backward()

            trainer.step(data.shape[0])

            train_loss += nd.mean(L).asscalar()
            ttrain_acc,ttrain_1acc,Predicted = accuracy(output, label)
            train_acc +=ttrain_acc
            if Predicted:
                train_1acc+=ttrain_1acc
                trueBatch+=1
                
            batch += 1
            if print_batches and batch % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f, Train True acc %f" % (
                    batch, train_loss/batch, train_acc/batch, ttrain_1acc / trueBatch
                ))
                
        test_acc, test_1acc = evaluate_accuracy(data_iter(X_test,y_test,batch_size), net, ctx)
        predict_acc, predict_1acc = evaluate_accuracy(data_iter(x_predict,y_predict,batch_size), net, ctx)
        
        print("Epoch %d. Loss: %f, Train acc: %f,Train True Value acc: %f, Test acc: %f, Test True Value acc: %f, Predict Acc: %f, Predict True Acc:%f" % (
            epoch, train_loss/batch, train_acc/batch,train_1acc / batch, test_acc, test_1acc,predict_acc,predict_1acc
        ))
        
        net.save_params("data/{}/{}-{}".format(name,name,str(epoch)))


def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        batch = 0.
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                L = loss(output, label)
            L.backward()

            trainer.step(data.shape[0])

            train_loss += nd.mean(L).asscalar()
            temp_acc, temp1_acc = accuracy(output, label)
            train_acc +=temp_acc
            
            batch += 1
            if print_batches and batch % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                    batch, train_loss/batch, train_acc/batch
                ))

        test_acc,test1_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f at %s" % (
            epoch, train_loss/batch, train_acc/batch, test_acc,datetime.now()
        ))
