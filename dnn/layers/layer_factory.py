import logging
from dnn.layers.linear_tanh import LinearTanh
from dnn.layers.linear_relu import LinearRelu
from dnn.layers.linear_leaky_relu import LinearLeakyRelu
from dnn.layers.linear_sigmoid import LinearSigmoid
from dnn.layers.linear_softmax import LinearSoftmax

class LayerFactory( ):
    @staticmethod
    def get_layer(h_l):
        logging.info('LayerFactory: ' + h_l)
        if h_l == "tanh":
            return LinearTanh, LinearSoftmax
        elif h_l == "relu":
            return LinearRelu, LinearSoftmax
        elif h_l == "reaky_relu":
            return LinearLeakyRelu, LinearSoftmax
        elif h_l == "sigmoid":
            return LinearSigmoid, LinearSoftmax
        else :
            return LinearRelu, LinearSoftmax
    