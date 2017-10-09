#encode=utf-8

from optimizer import Optimizer

class SgdOptimizer(Optimizer):
    def __init__(self, learning_rate):
        self._learning_rate = learning_rate
        
    def update_parameters(self, parameters, grads, L):
        for l in range(L):
            l = str(l + 1)
            parameters['W' + l] = parameters['W' + l] - self._learning_rate * grads['dW' + l]
            parameters['b' + l] = parameters['b' + l] - self._learning_rate * grads['db' + l]
        return parameters