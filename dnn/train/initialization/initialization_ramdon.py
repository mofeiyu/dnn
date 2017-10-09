import numpy as np

def initialize_ramdon(layers_sizes):
    parameters = {}
    layers_sizes.insert(0, 784) 
    L = len(layers_sizes)
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layers_sizes[l],layers_sizes[l-1])/10
        parameters['b' + str(l)] = np.zeros((layers_sizes[l],1))
    return parameters
