#encode=utf-8

import numpy as np
from dnn.optimizers.optimizer import Optimizer

class RmspropOptimizer(Optimizer):
    def __init__(self, conf, layer_dims):
        self._learning_rate = conf.learning_rate
        self._beta = conf.rmsprop.beta
        self._epsilon = conf.rmsprop.epsilon
        self._initialization(layer_dims)
    
    def _initialization(self, layer_dims):
        layers_number = len(layer_dims)
        self._v = {}
        for l in range(0, layers_number - 1):
            self._s["dW" + str(l + 1)] = np.zeros((layer_dims[l + 1], layer_dims[l]))
            self._s["db" + str(l + 1)] = np.zeros((layer_dims[l + 1], 1))
            
    def update_parameters(self, parameters, grads, L):
        for l in range(L):
            dW = 'dW' + str(l + 1)
            db = 'db' + str(l + 1)
            # update v
            self._s[dW] = self._beta * self._s[dW] + (1 - self._beta) * (grads[dW]**2)
            self._s[db] = self._beta * self._s[db] + (1 - self._beta) * (grads[db]**2)
            # update parameters
            parameters['W' + str(l+1)] -= self._learning_rate * dW/(self._s[dW]**0.5 + self._epsilon)
            parameters['b' + str(l+1)] -= self._learning_rate * db/(self._s[db]**0.5 + self._epsilon)
        return parameters