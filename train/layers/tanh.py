from basic_layer import Layer
class tanh(Layer):
    @staticmethod
    def forward(z):
        t = 2 /(1+np.exp(-2*z))  - 1  
        return t
    @staticmethod
    def backprop(dA):
        pass