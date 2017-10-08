from layers import softmax

def forward(config, act_layer, X, parameters, L):
    caches = []
    A = X
    print "A", A
    print act_layer
    for l in range(1, L - 1):
        A, cache = act_layer.forward(A, parameters['W' + str(l)], parameters['b' + str(l)])
        caches.append(cache)
    AL, cache = softmax.softmax.forward(A,parameters['W' + str(L - 1)], parameters['b' + str(L - 1)])  
    caches.append(cache)
    return AL
