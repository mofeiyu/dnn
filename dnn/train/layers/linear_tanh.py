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
    def forward(A,W,b):
        Z, cache = LinearLayer.forward(A,W,b)
        t = LinearTanh.tanh(Z)
        return t, cache
    @staticmethod
    def backward(dA, cache, c_f_type, lambd = 0):
        dZ = LinearTanh.tanh_backward(dA, cache)
        dA, dW, db = LinearLayer.backward(dZ, cache, c_f_type, lambd)
        return dA, dW, db