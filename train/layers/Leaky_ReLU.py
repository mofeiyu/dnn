import numpy as np
from basic_layer import Layer
class Leaky_ReLU(Layer):
    @staticmethod
    def forward(z):
        l = max(0.1*z,z)
        return l
    @staticmethod
    def backprop(dA):
        pass