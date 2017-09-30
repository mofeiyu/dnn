def initialize_ramdon(layers_sizes):
    parameters = {}
    L = len(layers_sizes)
    
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
    return parameters
