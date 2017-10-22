import numpy as np
from dnn.layers.basic_layer import Layer
from dnn.layers.linear import LinearLayer

class LinearTanh(Layer):
    @staticmethod
    def tanh(z): 
        return 2 /(1+np.exp(-2*z))  - 1 
    @staticmethod
    def tanh_backward(dA, cache):
        A, W, b= cache
        A, cache= LinearTanh.forward(A, W, b)
        dZ = np.multiply(dA, (1-np.multiply(A, A)))
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