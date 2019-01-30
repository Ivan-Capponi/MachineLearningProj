# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 23:06:39 2019

@author: Ivan Capponi
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import empty

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def neuralnet_initialization(x,y):
    return {'w1':np.random.rand(x.shape[1],10), 'w2':np.random.rand(10,1), 'out':np.zeros(y.shape)};    

def neuralnet_feedforward(x,w1,w2):
    layer = sigmoid(np.dot(x, w1));
    output = sigmoid(np.dot(layer, w2));
    return {'layer':layer, 'out':output}

def neuralnet_backpropagation(x, layer, prediction, y, w1, w2):
    tmp_w2 = np.dot(layer.T, (2*(y - prediction) * sigmoid_derivative(prediction)));
    tmp_w1 = np.dot(x.T, (np.dot(2*(y - prediction) * sigmoid_derivative(prediction), w2.T) * sigmoid_derivative(layer)));
    neww1 = w1 + tmp_w1
    neww2 = w2 + tmp_w2
    return {'w1': neww1, 'w2': neww2 }

def generate_dataset():
    arrList = empty([10,5])
    testList = empty([10,1])
    for i in range (0,10):
        arr = np.random.randint(2, size=5)
        res = 1
        for j in range(0,5):
            res = res + arr[j]
        testList[i][0] = res%2
        arrList[i] = (arr)
    return {'train': arrList, 'test': testList}
      
data = generate_dataset()
X = np.asarray(data['train'])
Y = np.asarray(data['test'])
nn = neuralnet_initialization(X,Y)
loss = []

for i in range(500):
    feed = neuralnet_feedforward(X,nn['w1'],nn['w2'])
    back = neuralnet_backpropagation(X, feed['layer'], feed['out'], Y, nn['w1'], nn['w2'])
    loss.append(np.sum(Y - feed['out'])**2) 
    nn['w1'] = back['w1']
    nn['w2'] = back['w2']
    nn['out'] = feed['out']
    

print(Y[:10])
print(nn['out'][:10])

t1 = np.arange(0.0, 200, 2)
plt.figure(1)
plt.plot(t1, loss[:100])
plt.show()