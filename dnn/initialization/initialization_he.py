import numpy as np

def initialization_he(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.multiply(np.random.randn(layers_dims[l],layers_dims[l-1]), np.sqrt(2.0 / layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
    return parameters