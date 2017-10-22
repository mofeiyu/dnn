import numpy as np
from dnn.layers.basic_layer import Layer
from dnn.layers.linear import LinearLayer
class LinearLeakyRelu(Layer):
    @staticmethod
    def leaky_relu(Z):
        r = Z > 0
        l = Z < 0
        l_r = l+r*0.1 
        return Z*l_r
    @staticmethod
    def leaky_relu_backward(dA,cache):
        A,W,b = cache
        Z, cache= LinearLayer.forward(A, W, b)
        r =  Z > 0
        l = (Z < 0) * 0.1
        l_r = r + l
        r = np.multiply(dA,l_r)
        return r
    @staticmethod
    def forward(A, W, b):
        Z, cache = LinearLayer.forward(A, W, b)
        s = LinearLeakyRelu.leaky_relu(Z)
        return s, cache
    @staticmethod
    def backward(dA, cache, c_f_type, lambd = 0):
        dZ = LinearLeakyRelu.leaky_relu_backward(dA,cache)
        dA, dW, db = LinearLayer.backward(dZ, cache, c_f_type, lambd)
        return dA, dW, db
