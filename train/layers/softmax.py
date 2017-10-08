import numpy as np
from dnn.train.layers.basic_layer import Layer
from dnn.train.layers.linear import LinearLayer
class softmax(Layer):
    @staticmethod
    def forward(A, W, b):
        Z, cache = LinearLayer.forward(A, W, b)
        S = np.exp(Z)/np.sum(np.exp(Z),axis = 1, keepdims = True)
        return S, cache
    @staticmethod
    def backward(dA, cache, l):
        dA, dW, db = LinearLayer.backward(dA, cache)
        return dA, dW, db