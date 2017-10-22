import numpy as np
from dnn.layers.basic_layer import Layer
from dnn.layers.linear import LinearLayer
class LinearSoftmax(Layer):
    @staticmethod
    def forward(A, W, b):
        Z, cache = LinearLayer.forward(A, W, b)
        S = np.exp(Z)/np.sum(np.exp(Z),axis = 0, keepdims = True)
        return S, cache
    @staticmethod
    def backward(dZ, cache,c_f_type, lambd = 0):
        dA, dW, db = LinearLayer.backward(dZ, cache, c_f_type, lambd)
        return dA, dW, db