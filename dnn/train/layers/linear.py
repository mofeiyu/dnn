import numpy as np
from basic_layer import Layer
class LinearLayer(Layer):
    @staticmethod
    def forward(A, W, b):
        Z = np.dot(W,A) + b
        cache = (A, W, b)
        return Z, cache

    @staticmethod
    def backward(dZ, cache):
        
        A_prev, W, b= cache
        m = A_prev.shape[1] 
        dW = np.dot(dZ, A_prev.T)*1/m
        db = np.sum(dZ, axis = 1, keepdims = True)*1/m
        dA_prev = np.dot(W.T, dZ)
  
        return dA_prev, dW, db
