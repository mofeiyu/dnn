import numpy as np
from basic_layer import Layer
from linear import LinearLayer
class LinearRelu(Layer):
    @staticmethod
    def relu(Z):
        temp = Z > 0
        return Z * temp
    @staticmethod
    def relu_backward(dA,cache):
        A, = cache
        r =  A > 0
        r = np.multiply(dA,r)
        return r
    @staticmethod
    def forward(A, W, b):
        print A
        print W
        print b
        Z, cache = LinearLayer.forward(A, W, b)
        s = LinearRelu.relu(Z)
        return s, cache
    @staticmethod
    def backward(dA, cache, l):
        dZ = LinearRelu.relu_backward(dA)
        dA, dW, db = LinearLayer.backward(dZ, cache)
        return dA, dW, db
