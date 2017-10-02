import numpy as np
from basic_layer import Layer
from linear import LinearLayer

class LinearTanh(Layer):
    @staticmethod
    def tanh(z): 
        return 2 /(1+np.exp(-2*z))  - 1 
    @staticmethod
    def tanh_backward(dA, cache):
        A, = cache
        dZ = np.dot(dA, (1-A**2))
        return dZ
    @staticmethod
    def forward(A):
        Z, cache = LinearLayer.forward(A)
        t = tanh(Z)
        return t, cache
    @staticmethod
    def backward(dA, cache, l):
        dZ = tanh_backward(dA, cache)
        dA, dW, db = LinearLayer.backward(dZ, cache)
        return dA, dW, db