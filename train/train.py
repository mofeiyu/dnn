# coding: utf-8

import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
import initialization

import backward
import forward
import get_

from dnn.load_data.normalization import normalization
from dnn.load_data import  mini_batch
from initialization import initialization_he,initialization_ramdon

def initialize_parameters(config):
    i_p = config.initialize_parameters
    if i_p == "initialization_ramdon":
        parameters = initialization_ramdon(config.layers_sizes)
    elif i_p == "initialization_he":
        parameters = initialization_he.initialization_he(config.layers_sizes)
    else:
        logging.error("Please choice a method of initialize_parameters for your model")
        sys.exit()
    return parameters

def L_layer_model(config, dataset, num_iterations = 3000, print_cost=False):
    X, Y = dataset.get_training_data()
    learning_rate = config.learning_rate
    L = len(config.layers_sizes)
    m = X.shape[1]
    print X.shape
    print Y.shape
    cost = []
    parameters = initialize_parameters(config)
    if config.data_stand == "normalization":
        X, mean, var = normalization(X)
        parameters['u'] = mean
        parameters['v'] = var
    batch_num = m / config.batch_size
    if m % config.batch_size != 0:
        batch_num = batch_num + 1
    
    act_layer, cost_function, update_parameters = get_.get(config)
    costs = []
    for i in range(0, num_iterations): 
        if batch_num != 1:
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation].reshape((10,m))           
        for j in range(0,batch_num): 
            if batch_num != 1:
                Xj,Yj = mini_batch.mini_batch_data(shuffled_X,shuffled_Y, j, config.batch_size, batch_num, config.data_stand)
            else:
                Xj,Yj = X,Y
            print "Xj", Xj
            AL, caches = forward.forward(config, act_layer, Xj, parameters, L)            
            cost = cost_function.cost(AL, Yj, L, parameters)            
            grads = backward.backward(config, cost, act_layer, caches)
            parameters = update_parameters.update_parameters(parameters, grads)
        
        logging.info("Cost after iteration %i: %f" %(i, cost))
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