import numpy as np
from basic_layer import Layer
from linear import LinearLayer
class LinearSigmoid(Layer):
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    @staticmethod
    def sigmoid_backward(dA, cache):
        A, W, b = cache
        dZ = np.dot(dA, np.dot(A, (1 - A)))
        return dZ
    @staticmethod
    def forward(A,W,b):
        Z, cache = LinearLayer.forward(A,W,b)
        s = LinearSigmoid.sigmoid(Z)
        return s, cache
    @staticmethod
    def backward(dA, cache, c_f_type, lambd = 0):
        dZ = LinearSigmoid.sigmoid_backward(dA, cache)
        dA, dW, db = LinearLayer.backward(dZ, cache, c_f_type, lambd)
        return dA, dW, db