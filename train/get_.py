import cost_function
import opt
import layers

from cost_function import basic_cost_function,l1_cost_function,l2_cost_function,svm_cost_function
from opt import adam,momentum,rmsprop,sgd
from layers import linear, linear_sigmoid, linear_relu, linear_tanh, softmax, linear, linear_leaky_relu, basic_layer

import sys
import logging

def get_layer(a_f):
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
        
def get_cost_funtion(c_f):
    if c_f == "lr_cost_function":
        return cost_function.CostFunction
    elif c_f == "l1_cost_function":
        return l1_cost_function.L1CostFunction
    elif c_f == "l2_cost_function":
        return l2_cost_function.L2CostFunction
    else:
        logging.error("Please choice a method of cost function for your model ")
        sys.exit()

def get_update_parameters(conf):
    L = len(conf.layers)
    learning_rate = conf.learning_rate
    opt = conf.opt
    if opt == "adam": 
        adam = adam.AdamOptimizer(L, learning_rate, conf.beta1, conf.beta2)             
        return adam
    elif opt == "momentum":
        mom = momentum.MomentumOptimizer(L, learning_rate)            
        return mom
    elif opt == "sgd":  
        sgd = sgd.SgdOptimizer(learning_rate)             
        return sgd
    elif opt == "rmsprop":  
        rms = rms.RmspropOptimizer(L, learning_rate)              
        return rms
    else:
        logging.error("Print choice a method of opt for your model")
        sys.exit()
        
        
def get(config):
    layer = get_layer(conf.activation_function)
    cost_fun = get_cost_funtion(config.cost_fun)
    opt = get_update_parameters(config)
    return layer, cost_fun, opt
