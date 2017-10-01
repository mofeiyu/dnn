import numpy as np
import layers
import logging

from layers import linear, linear_sigmoid, linear_relu, linear_tanh, softmax, linear, linear_leaky_relu, basic_layer


def backward(config, act_layer, cost, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Yj.reshape(AL.shape)
    dAL = cost
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = softmax.Softmax.backward(dAL , current_cache)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = act_layer.backward(grads["dA" + str(l + 2)], current_cache)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads
