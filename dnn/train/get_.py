import cost_function
import opt
import layers

from dnn.train.cost_function import basic,cost_function,l1_cost_function,l2_cost_function
from dnn.train.opt import adam,momentum,rmsprop,sgd
from dnn.train.layers import linear, linear_sigmoid, linear_relu, linear_tanh, softmax, linear, linear_leaky_relu, basic_layer

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
    if c_f == "cost_function":
        return cost_function.CostFunction
    elif c_f == "l1_cost_function":
        return l1_cost_function.L1CostFunction
    elif c_f == "l2_cost_function":
        return l2_cost_function.L2CostFunction
    else:
        logging.error("Please choice a method of cost function for your model ")
        sys.exit()

def get_update_parameters(conf):
    L = len(conf.layers_sizes)
    learning_rate = conf.learning_rate
    opt = conf.opt
    if opt == "adam": 
        return adam.AdamOptimizer(conf.layers_sizes, learning_rate, conf.adam.beta1, conf.adam.beta2)             
    elif opt == "momentum":
        return momentum.MomentumOptimizer(conf.layers_sizes, learning_rate)            
    elif opt == "sgd":  
        return sgd.SgdOptimizer(learning_rate)             
    elif opt == "rmsprop":  
        return rmsprop.RmspropOptimizer(conf.layers_sizes, learning_rate)              
    else:
        logging.error("Print choice a method of opt for your model")
        sys.exit()
        
def get(config):
    layer = get_layer(config.activation_function)
    cost_fun = get_cost_funtion(config.cost_fun)
    opt = get_update_parameters(config)
    return layer, cost_fun, opt
