#encode=utf-8

import numpy as np
import optimizer
from optimizer import Optimizer

class MomentumOptimizer(Optimizer):
    def __init__(self, layers_dims, learning_rate=0.001, beta=0.9):
        self._learning_rate = learning_rate
        self._beta = beta
        self._initialization(layers_dims)
    
    def _initialization(self, layers_dims):
        layers_number = len(layers_dims)
        self._v = {}
        for l in range(0, layers_number):
            self._v["dW" + str(l + 1)] = np.zeros((layers_dims[l + 1], layers_dims[l]))
            self._v["db" + str(l + 1)] = np.zeros((layers_dims[l + 1], 1))
            
    def update_parameters(self, parameters, grads):
        layers_number = len(parameters)
        for l in range(layers_number):
            dW = 'dW' + str(l + 1)
            db = 'db' + str(l + 1)
            # update v
            self._v[dW] = self._beta * self._v[dW] + (1 - self._beta) * grads[dW]
            self._v[db] = self._beta * self._v[db] + (1 - self._beta) * grads[db]
            # update parameters
            parameters['W' + l] -= self._learning_rate * self._v[dW]
            parameters['b' + l] -= self._learning_rate * self._beta * self._v[db]
            