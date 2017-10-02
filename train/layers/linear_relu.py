from numpy import np
from linear import LinearLayer
class LinearRelu(Layer):
    @staticmethod
    def relu(Z):
        r = max(0,Z) 
        return r
    @staticmethod
    def relu_backward(dA,cache):
        A, = cache
        r =  A > 0
        r = np.multiply(dA,r)
        return r
    def forward(A):
        Z, cache = LinearLayer.forward(A)
        s = relu(Z)
        return s, cache
    @staticmethod
    def backward(dA, cache, l):
        dZ = relu_backward(dA)
        dA, dW, db = LinearLayer.backward(dZ, cache)
        return dA, dW, db
