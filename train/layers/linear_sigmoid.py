from basic_layer import Layer
from linear import LinearLayer
class Sigmoid(Layer):
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    @staticmethod
    def forward(a):
        z = LinearLayer.forward(a)
        s = sigmoid(z)
        return s
    @staticmethod
    def backprop(dA):
        dZ = sigmoid(z) * (1 - sigmoid(z))
        pass