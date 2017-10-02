#encode=utf-8

import numpy as np
import optimizer
from optimizer import Optimizer

class AdamOptimizer(Optimizer):
    def __init__(self, layers_dims, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._t = 0
        layers_number = len(layers_dims)
        self._v = {}
        self._s = {}
        for l in range(0, layers_number):
            self._v["dW" + str(l + 1)] = np.zeros((layers_dims[l + 1], layers_dims[l]))
            self._v["db" + str(l + 1)] = np.zeros((layers_dims[l + 1], 1))
            self._s["dW" + str(l + 1)] = np.zeros((layers_dims[l + 1], layers_dims[l]))
            self._s["db" + str(l + 1)] = np.zeros((layers_dims[l + 1], 1))
    
    def update_parameters(self, parameters, grads):
        self._t += 1
        layers_number = len(parameters)
        for l in range(layers_number):
            dW = 'dW' + str(l + 1)
            db = 'db' + str(l + 1)
            # update v
            self._v[dW] = self._beta1 * self._v[dW] + (1 - self._beta1) * grads[dW]
            self._v[db] = self._beta1 * self._v[db] + (1 - self._beta1) * grads[db]
            v_dW_corrected = self._v[dW] / (1 - self._beta1 ** self._t)
            v_db_corrected = self._v[db] / (1 - self._beta1 ** self._t)
            # update s
            self._s[dW] = self._beta2 * self._s[dW] + (1 - self._beta2) * (grads[dW] ** 2)
            self._s[db] = self._beta2 * self._s[db] + (1 - self._beta2) * (grads[db] ** 2)
            s_dW_corrected = self._s[dW] / (1 - self._beta2 ** self._t)
            s_db_corrected = self._s[db] / (1 - self._beta2 ** self._t)
            # update parameters
            parameters['W' + l] -= self._learning_rate * v_dW_corrected / (s_dW_corrected ** 0.5 + self._epsilon)
            parameters['b' + l] -= self._learning_rate * v_db_corrected / (s_db_corrected ** 0.5 + self._epsilon)