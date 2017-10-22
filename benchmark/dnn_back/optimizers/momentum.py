#encode=utf-8

import numpy as np
from dnn.optimizers.optimizer import Optimizer

class MomentumOptimizer(Optimizer):
    def __init__(self, conf, layer_dims):
        self._learning_rate = conf.learning_rate
        self._beta = conf.momentum.beta
        self._initialization(layer_dims)
    
    def _initialization(self, layer_dims):
        layers_number = len(layer_dims)
        self._v = {}
        for l in range(0, layers_number - 1):
            self._v["dW" + str(l + 1)] = np.zeros((layer_dims[l + 1], layer_dims[l]))
            self._v["db" + str(l + 1)] = np.zeros((layer_dims[l + 1], 1))
            
    def update_parameters(self, parameters, grads, L):
        for l in range(L):
            dW = 'dW' + str(l + 1)
            db = 'db' + str(l + 1)
            # update v
            self._v[dW] = self._beta * self._v[dW] + (1 - self._beta) * grads[dW]
            self._v[db] = self._beta * self._v[db] + (1 - self._beta) * grads[db]
            # update parameters
            parameters['W' + str(l+1)] -= self._learning_rate * self._v[dW]
            parameters['b' + str(l+1)] -= self._learning_rate * self._beta * self._v[db]
        return parameters   