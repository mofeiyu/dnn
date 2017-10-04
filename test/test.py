import numpy as np

def test(train_x, train_y, parameters):
    L = len(parameters) // 2  
    for l in range(1, L):
        A, cache = act_layer.forward(train_x, parameters)
    AL = softmax.forward(A) 
    
    return 