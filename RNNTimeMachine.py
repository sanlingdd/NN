# coding=utf-8
import sys
sys.path.append('..')

from mxnet import ndarray as nd
import re

with open("data/timemachine.txt") as f:
    time_machine = f.read()

def getWords(string):
    eraseString = string.lower().replace('\n', '').replace('\r', '').replace('\s','')
    return re.split('(\W)', eraseString)

#time_machine = time_machine.lower().replace('\n', '').replace('\r', '').replace('\s','')
time_machine = getWords(time_machine)

character_list = list(set(time_machine))
character_dict = dict([(char,i) for i,char in enumerate(character_list)])

vocab_size = len(character_dict)

print('vocab size:', vocab_size)

time_numerical = [character_dict[char] for char in time_machine]

import random
from mxnet import nd

def data_iter(batch_size, seq_len, ctx=None):
    num_examples = (len(time_numerical)-1) // seq_len
    num_batches = num_examples // batch_size
    # ?????
    idx = list(range(num_examples))
    random.shuffle(idx)
    # ??seq_len???
    def _data(pos):
        return time_numerical[pos:pos+seq_len]
    for i in range(num_batches):
        # ????batch_size?????
        examples = idx[i:i+batch_size]
        data = nd.array(
            [_data(j*seq_len) for j in examples], ctx=ctx)
        label = nd.array(
            [_data(j*seq_len+1) for j in examples], ctx=ctx)
        yield data, label

import mxnet as mx

#GPU
import sys
sys.path.append('..')
import utils
ctx = utils.try_gpu()
print('Will use ', ctx)

num_hidden = 256
weight_scale = .01

# ???
Wxh = nd.random_normal(shape=(vocab_size,num_hidden), ctx=ctx) * weight_scale
Whh = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx) * weight_scale
bh = nd.zeros(num_hidden, ctx=ctx)
# ???
Why = nd.random_normal(shape=(num_hidden,vocab_size), ctx=ctx) * weight_scale
by = nd.zeros(vocab_size, ctx=ctx)

params = [Wxh, Whh, bh, Why, by]
for param in params:
    param.attach_grad()


def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]

def rnn(inputs, H):
    # inputs: seq_len ? batch_size x vocab_size ??
    # H: batch_size x num_hidden ??
    # outputs: seq_len ? batch_size x vocab_size ??
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, Wxh) + nd.dot(H, Whh) + bh)
        Y = nd.dot(H, Why) + by
        outputs.append(Y)
    return (outputs, H)



def predict(prefix, num_chars):
    # ??? prefix ??????? num_chars ???
    prefix = getWords(prefix)
    state = nd.zeros(shape=(1, num_hidden), ctx=ctx)
    output = [character_dict[prefix[0]]]
    for i in range(num_chars+len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        Y, state = rnn(get_inputs(X), state)
        #print(Y)
        if i < len(prefix)-1:
            next_input = character_dict[prefix[i+1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([character_list[i] for i in output])


def grad_clipping(params, theta):
    norm = nd.array([0.0], ctx)
    for p in params:
        norm += nd.sum(p.grad ** 2)
    norm = nd.sqrt(norm).asscalar()
    if norm > theta:
        for p in params:
            p.grad[:] *= theta/norm


from mxnet import autograd
from mxnet import gluon
from math import exp

epochs = 200
seq_len = 35
learning_rate = .1
batch_size = 32

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

for e in range(epochs+1):
    train_loss, num_examples = 0, 0
    state = nd.zeros(shape=(batch_size, num_hidden), ctx=ctx)
    for data, label in data_iter(batch_size, seq_len, ctx):
        with autograd.record():
            outputs, state = rnn(get_inputs(data), state)
            # reshape label to (batch_size*seq_len, )
            # concate outputs to (batch_size*seq_len, vocab_size)
            label = label.T.reshape((-1,))
            outputs = nd.concat(*outputs, dim=0)
            loss = softmax_cross_entropy(outputs, label)
        loss.backward()

        grad_clipping(params, 5)
        utils.SGD(params, learning_rate)

        train_loss += nd.sum(loss).asscalar()
        num_examples += loss.size

    if e % 20 == 0:
        print("Epoch %d. PPL %f" % (e, exp(train_loss/num_examples)))
        print(' - ', predict('The Time ', 100))
        print(' - ', predict("The Medical Man rose, came to the lamp,", 100), '\n')
        