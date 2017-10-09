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
        A, W, b = cache
        Z, cache= LinearLayer.forward(A, W, b)
        r =  Z > 0
        r = np.multiply(dA,r)
        return r
    @staticmethod
    def forward(A, W, b):
        Z, cache = LinearLayer.forward(A, W, b)
        s = LinearRelu.relu(Z)
        return s, cache
    @staticmethod
    def backward(dA, cache):
        dZ = LinearRelu.relu_backward(dA,cache)
        dA, dW, db = LinearLayer.backward(dZ, cache)
        return dA, dW, db
