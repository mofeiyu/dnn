from numpy import np
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
    def forward(A):
        Z, cache = LinearLayer.forward(A)
        s = leaky_relu(Z)
        return s, cache
    @staticmethod
    def backward(dA, cache, l):
        dZ = leaky_relu_backward(dA)
        dA, dW, db = LinearLayer.backward(dZ, cache)
        return dA, dW, db

    
