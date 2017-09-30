from basic_layer import Layer
class ReLU(Layer):
    @staticmethod
    def forward(z):
        r = max(0,z) 
        return r
    @staticmethod
    def backprop(dA):
        pass