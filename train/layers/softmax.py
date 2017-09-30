import numpy as np
from basic_layer import Layer
class softmax(Layer):
    @staticmethod
    def forward(z):
        s = np.exp(z)/np.sum(np.exp(z),axis = 1)
        return s
    @staticmethod
    def backprop(dA):
        pass