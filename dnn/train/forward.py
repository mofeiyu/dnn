from layers import softmax

def forward(config, act_layer, X, parameters, L):
    caches = []
    A = X
    for l in range(1, L):
        A, cache = act_layer.forward(A, parameters['W' + str(l)], parameters['b' + str(l)])
        caches.append(cache)
    AL, cache = softmax.softmax.forward(A,parameters['W' + str(L)], parameters['b' + str(L)])  
    caches.append(cache)
    return AL, caches
