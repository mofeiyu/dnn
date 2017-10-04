# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import initialization


import logging
import backward
import fordward
import data
import get_

from data import min_batch, normalization
from initialization import initialization_he,initialization_ramdon


def initialize_parameters(i_p):
    if i_p == "initialization_ramdon":
        parameters = initialization_ramdon.initialization_ramdon(config.layers_sizes)
    elif i_p == "initialization_he":
        parameters = initialization_he.initialization_he(config.layers_sizes)
    else:
        logging.error("Please choice a method of initialize_parameters for your model")
        sys.exit()
    return parameters

def L_layer_model(config, X, Y, num_iterations = 3000, print_cost=False):        
    
    learning_rate = config.learn_rate
    L = len(config.layers_sizes)
    m = X.shape[1]
    cost = []
    parameters = initialize_parameters(config.initialize_parameters)
    if config.data_stand == "normalization":
        X, mean, var = normalization.normalization(X)
        parameters['u'] = mean
        parameters['v'] = var
        
    batch_num = m / config.batch_size
    if m % config.batch_size != 0:
        batch_num == batch_num + 1
    
    act_layer, cost_function, update_parameters = get_.get(config)
        
    for i in range(0, num_iterations): 
        if batch_num != 1:
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation].reshape((1,m))           
        for j in rang(0,batch_num): 
            if batch_num != 1:
                Xj,Yj = mini_batch.mini_batch_data(shuffled_X,shuffled_Y, j, config.batch_size, batch_num, config.data_stand)
            else:
                Xj,Yj = X,Y   
                          
            AL, caches = forward.forward(config, act_layer, Xj, parameters)            
            cost = cost_funtion.cost(AL, Yj, L, parameters)            
            grads = backward.backward(config, cost, act_layer, caches)
            parameters = update_parameters.update_parameters(parameters, grads)
            
        if print_cost and i % 100 == 0:
           logging.info("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters