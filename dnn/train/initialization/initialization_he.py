import numpy as np

def initialization_he(layers_sizes):
    parameters = {}
    layers_sizes.insert(0, 784)
    L = len(layers_sizes)
    for l in range(1, L):
        parameters['W' + str(l)] = np.multiply(np.random.randn(layers_sizes[l],layers_sizes[l-1]), np.sqrt(2.0 / layers_sizes[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_sizes[l],1))
    return parameters