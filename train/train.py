# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import initialization
import cost_function
import opt

import backward
import fordward

from initialization import initialization_he,initialization_ramdon
from cost_function import basic_cost_function,l1_cost_function,l2_cost_function,svm_cost_function
from opt import adam,momentum,rmsprop,sgd

def L_layer_model(config, X, Y, num_iterations = 3000, print_cost=False):#
    learning_rate = config.learn_rate
    m = X.shape[1]
    cost = []
    if config.initialize_parameters == "initialization_ramdon":
        parameters = initialization_ramdon.initialization_ramdon(config.layers_sizes)
    elif config.initialize_parameters == "initialization_he":
        parameters = initialization_he.initialization_he(config.layers_sizes)
    else:
        print "Please choice a method of initialize_parameters for your model"
        break
    

    batch_num = m / config.batch_size
    if m % config.batch_size != 0:
        batch_num == batch_num + 1
    
    for i in range(0, num_iterations): 
        if batch_num != 1:
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation].reshape((1,m))            
        for j in rang(0,batch_num): 
            if batch_num != 1:
                if j != batch_num-1:
                    Xj,Yj = X[:,j*config.batch_size:(j+1)*config.batch_size],Y[:,j*config.batch_size:(j+1)*config.batch_size]
                    if config.data_stand == "batch_normalization":
                        Xj = normalization.normalization(Xj)
                else :
                    Xj,Yj = X[:,j*config.batch_size:],Y[:,j*config.batch_size:]
                    if config.data_stand == "batch_normalization":
                        Xj = normalization.normalization(Xj)                    
            
            AL, caches = forward.forward(config, Xj, parameters)
            
            if config.cost_fun == "lr_cost_function":
                cost = lr_cost_function.lr_cost_function(AL, Yj)
            elif config.cost_fun == "l1_cost_function":
                cost = l1_cost_function.l1_cost_function(AL, Yj)
            elif config.cost_fun == "l2_cost_function":
                cost = l2_cost_function.l2_cost_function(AL, Yj)
            elif config.cost_fun == "svm_cost_function":
                cost = svm_cost_function.svm_basic_cost_function(AL, Yj)
            else:
                print "Please choice a method of cost function for your model "
                break   
            
            grads = backward.backward(config, AL, Yj, caches)
            
            if config.opt == "adam":                
                parameters = adam.adam_update_parameters(parameters, grads, learning_rate)
            elif config.opt == "momentum":                
                parameters = momentum.momentum_update_parameters(parameters, grads, learning_rate)
            elif config.opt == "sgd":                
                parameters = sgd.sgd_update_parameters(parameters, grads, learning_rate)
            elif config.opt == "rmsprop":                
                parameters = rmsprop.rmsprop_update_parameters(parameters, grads, learning_rate)
            else:
                print "Print choice a method of opt for your model"
                break
            
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