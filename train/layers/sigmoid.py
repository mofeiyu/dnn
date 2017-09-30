from basic_layer import Layer
class Sigmoid(Layer):
    @staticmethod
    def forward(z):
        s = 1/(1+np.exp(-z))    
        return s
    @staticmethod
    def backprop(dA):
        pass