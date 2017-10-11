#encode=utf-8

from dnn.optimizers.optimizer import Optimizer

class SgdOptimizer(Optimizer):
    def __init__(self, conf, layer_dims):
        self._learning_rate = conf.learning_rate
        
    def update_parameters(self, parameters, grads, L):
        for l in range(L):
            l = str(l + 1)
            parameters['W' + str(l)] = parameters['W' + str(l)] - self._learning_rate * grads['dW' + str(l)]
            parameters['b' + str(l)] = parameters['b' + str(l)] - self._learning_rate * grads['db' + str(l)]
        return parameters