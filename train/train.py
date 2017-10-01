# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import initialization
import cost_function
import opt
import sys

import logging
import backward
import fordward
import layers

from initialization import initialization_he,initialization_ramdon
from cost_function import basic_cost_function,l1_cost_function,l2_cost_function,svm_cost_function
from opt import adam,momentum,rmsprop,sgd
from layers import linear, linear_sigmoid, linear_relu, linear_tanh, softmax, linear, linear_leaky_relu, basic_layer

def get_layer(config):
    a_f = config.layer
    if a_f == "relu":
        return linear_relu.LinearRelu
    elif a_f == "sigmoid":
        return linear_sigmoid.LinearSigmoid
    elif a_f == "tanh":
        return linear_tanh.LinearTanh
    elif a_f == "leaky_relu":
        return linear_leaky_relu.LinearLeakyRelu
    else:
        logging.error("Please choice a function for your layer ")
        sys.exit()

def initialize_parameters(i_p):
    if i_p == "initialization_ramdon":
        parameters = initialization_ramdon.initialization_ramdon(config.layers_sizes)
    elif i_p == "initialization_he":
        parameters = initialization_he.initialization_he(config.layers_sizes)
    else:
        logging.error("Please choice a method of initialize_parameters for your model")
        sys.exit()
    return parameters

def mini_batch_data(X,Y,j,batch_size,batch_num,data_stand):
    if j != batch_num-1:
        Xj,Yj = X[:,j*batch_size:(j+1)*batch_size],Y[:,j*batch_size:(j+1)*batch_size]
        if data_stand == "batch_normalization":
            Xj = normalization.normalization(Xj)
    else :
        Xj,Yj = X[:,j*batch_size:],Y[:,j*batch_size:]
        if data_stand == "batch_normalization":
            Xj = normalization.normalization(Xj) 
    return Xj,Yj   

def cost_funtion(config, AL, Yj):
    cost_fun = config.cost_fun
    if cost_fun == "lr_cost_function":
        cost = lr_cost_function.lr_cost_function(AL, Yj)
    elif cost_fun == "l1_cost_function":
        cost = l1_cost_function.l1_cost_function(AL, Yj, parameters)
    elif cost_fun == "l2_cost_function":
        cost = l2_cost_function.l2_cost_function(AL, Yj, parameters)
    elif cost_fun == "svm_cost_function":
        cost = svm_cost_function.svm_basic_cost_function(AL, Yj)
    else:
        logging.error("Please choice a method of cost function for your model ")
        sys.exit()
    return cost

def update_parameters(config,  parameters, grads, learning_rate):
    opt = config.opt
    if opt == "adam":                
        parameters = adam.adam_update_parameters(config,parameters, grads, learning_rate)
    elif opt == "momentum":                
        parameters = momentum.momentum_update_parameters(config,parameters, grads, learning_rate)
    elif opt == "sgd":                
        parameters = sgd.sgd_update_parameters(config,parameters, grads, learning_rate)
    elif opt == "rmsprop":                
        parameters = rmsprop.rmsprop_update_parameters(config,parameters, grads, learning_rate)
    else:
        logging.error("Print choice a method of opt for your model")
        sys.exit()
    return  parameters 

def L_layer_model(config, X, Y, num_iterations = 3000, print_cost=False):#
    learning_rate = config.learn_rate
    m = X.shape[1]
    cost = []
    parameters = initialize_parameters(config.initialize_parameters)

    batch_num = m / config.batch_size
    if m % config.batch_size != 0:
        batch_num == batch_num + 1
    
    act_layer = get_layer(config)
    
    for i in range(0, num_iterations): 
        if batch_num != 1:
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation].reshape((1,m))           
        for j in rang(0,batch_num): 
            if batch_num != 1:
                Xj,Yj = mini_batch_data(shuffled_X,shuffled_Y, j, config.batch_size, batch_num, config.data_stand)
            else:
                Xj,Yj = X,Y   
                          
            AL, caches = forward.forward(config, act_layer, Xj, parameters)            
            cost = cost_funtion(config, AL, Yj)            
            grads = backward.backward(config, act_layer, AL, Yj, caches)
            parameters = update_parameters(config, parameters, grads, learning_rate)
            
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters