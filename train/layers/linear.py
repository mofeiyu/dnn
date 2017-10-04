import numpy as np
from basic_layer import Layer
class LinearLayer(Layer):
    @staticmethod
    def linear_forward(A, W, b):
        Z = np.dot(W,A) + b
        cache = (A, W, b)   
        return Z, cache

    @staticmethod
    def backward(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1] 
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)    
        return dA_prev, dW, db
