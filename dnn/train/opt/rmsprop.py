#encode=utf-8

import numpy as np
import optimizer
from optimizer import Optimizer

class RmspropOptimizer(Optimizer):
    def __init__(self, layers_dims, learning_rate=0.001, beta=0.999, epsilon=1e-8):
        self._learning_rate = learning_rate
        self._beta = beta
        self._epsilon = epsilon
        self._initialization(layers_dims)
    
    def _initialization(self, layers_dims):
        layers_number = len(layers_dims)
        self._v = {}
        for l in range(0, layers_number):
            self._s["dW" + str(l + 1)] = np.zeros((layers_dims[l + 1], layers_dims[l]))
            self._s["db" + str(l + 1)] = np.zeros((layers_dims[l + 1], 1))
            
    def update_parameters(self, parameters, grads):
        layers_number = len(parameters)
        for l in range(layers_number):
            dW = 'dW' + str(l + 1)
            db = 'db' + str(l + 1)
            # update v
            self._s[dW] = self._beta * self._s[dW] + (1 - self._beta) * (grads[dW]**2)
            self._s[db] = self._beta * self._s[db] + (1 - self._beta) * (grads[db]**2)
            # update parameters
            parameters['W' + l] -= self._learning_rate * dW/(self._s[dW]**0.5 + self._epsilon)
            parameters['b' + l] -= self._learning_rate * db/(self._s[db]**0.5 + self._epsilon)
  