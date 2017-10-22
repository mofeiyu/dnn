#encode=utf-8

import numpy as np
from dnn.optimizers.optimizer import Optimizer

class AdamOptimizer(Optimizer):
    def __init__(self, conf, layer_dims):
        self._learning_rate = conf.learning_rate
        self._beta1 = conf.adam.beta1
        self._beta2 = conf.adam.beta2
        self._epsilon = conf.adam.epsilon
        self._t = 0
        layers_number = len(layer_dims)
        self._v = {}
        self._s = {}
        for l in range(0, layers_number - 1):
            self._v["dW" + str(l + 1)] = np.zeros((layer_dims[l + 1], layer_dims[l]))
            self._v["db" + str(l + 1)] = np.zeros((layer_dims[l + 1], 1))
            self._s["dW" + str(l + 1)] = np.zeros((layer_dims[l + 1], layer_dims[l]))
            self._s["db" + str(l + 1)] = np.zeros((layer_dims[l + 1], 1))
    
    def dump(self):
        return {
            'opt': 'adam',
            'beta1': self._beta1,
            'beta2': self._beta2,
            'epsilon': self._epsilon,
            't': self._t,
            'v': self._v,
            's': self._s
        }
    
    def load(self, cache_data_dict, learning_rate):
        self._learning_rate = learning_rate
        self._beta1 = cache_data_dict['beta1']
        self._beta2 = cache_data_dict['beta2']
        self._epsilon = cache_data_dict['epsilon']
        self._t = cache_data_dict['t']
        self._v = cache_data_dict['v']
    
    def update_parameters(self, parameters, grads, L):
        self._t += 1
        for l in range(L):
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
            parameters['W' + str(l+1)] -= self._learning_rate * v_dW_corrected / (s_dW_corrected ** 0.5 + self._epsilon)
            parameters['b' + str(l+1)] -= self._learning_rate * v_db_corrected / (s_db_corrected ** 0.5 + self._epsilon)
        return parameters
