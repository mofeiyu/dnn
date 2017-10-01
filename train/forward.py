import numpy as np
import layers
import logging

from layers import linear, linear_sigmoid, linear_relu, linear_tanh, softmax, linear, linear_leaky_relu, basic_layer

def forward(config, act_layer, Xj, parameters):
    caches = []
    L = len(parameters) // 2  
    A = Xj
    for l in range(1, L):
        A, cache = act_layer.forward(A, parameters)
        caches.append(cache)
    AL,cache = softmax.forward(A)    
    return AL, caches