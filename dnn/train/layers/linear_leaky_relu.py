import numpy as np
from basic_layer import Layer
from linear import LinearLayer
class LinearLeakyRelu(Layer):
    @staticmethod
    def leaky_relu(Z):
        r = max(0.1*Z,Z) 
        return r
    @staticmethod
    def leaky_relu_backward(dA,cache):
        A, = cache
        r =  A > 0
        l = (A < 0) * 0.1
        l_r = r + l
        r = np.multiply(dA,l_r)
        return r
    @staticmethod
    def forward(A, W, b):
        Z, cache = LinearLayer.forward(A, W, b)
        s = LinearLeakyRelu.leaky_relu(Z)
        return s, cache
    @staticmethod
    def backward(dA, cache):
        dZ = LinearLeakyRelu.leaky_relu_backward(dA)
        dA, dW, db = LinearLayer.backward(dZ, cache)
        return dA, dW, db
