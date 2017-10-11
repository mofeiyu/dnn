#encode=utf-8

import os
import json
import logging

def singleton(cls):
    instance = cls()
    instance.__call__ = lambda: instance
    return instance

class LayerConfig:
    def __init__(self, conf):
        self.hidden_units = conf['hidden_units']
        self.dropout = conf['dropout']
        self.activation_function = conf['activation_function']
        self.input_size = conf['input_size']
        self.n_classes = conf['n_classes']

class MomentumConfig:
    def __init__(self, conf):
        self.beta = 0.9 if 'beta' not in conf else conf['beta']

class RmspropConfig:
    def __init__(self, conf):
        self.beta = 0.999 if 'beta' not in conf else conf['beta']
        self.epsilon = 1e-8 if 'epsilon' not in conf else conf['epsilon']

class AdamConfig:
    def __init__(self, conf):
        self.beta1 = 0.9 if 'beta1' not in conf else conf['beta1']
        self.beta2 = 0.999 if 'beta2' not in conf else conf['beta2']
        self.epsilon = 1e-8 if 'epsilon' not in conf else conf['epsilon']

class CostFunction:
    def __init__(self, conf):
        self.cost_function = conf['cost_function']
        if conf["cost_function"] == "log-likelihood":
            self.regularization_strength = 0
        elif conf["cost_function"] == "l1_log-likelihood":
            self.regularization_strength = conf['l1_regularization_strength']
        elif conf["cost_function"] == "l2_log-likelihood":
            self.regularization_strength = conf['l2_regularization_strength']
@singleton
class Config():
    def __init__(self):
        conf_path = os.path.dirname(os.path.abspath(__file__)) + '/conf.json'
        conf = json.load(file(conf_path))
        self.normalization = conf["normalization"]
        self.initialization = conf['initialization']
        self.learning_rate = conf['learning_rate']
        self.learning_decay = conf['learning_decay']
        self.batch_size = conf['batch_size']
        self.optimizer = conf['optimizer']
        self.checkpoint = conf['checkpoint']
        self.init_checkpoint = conf['init_checkpoint']
        self.adam = AdamConfig(conf['adam'])
        self.momentum = MomentumConfig(conf['momentum'])
        self.rmsprop = RmspropConfig(conf['rmsprop'])
        self.adam = AdamConfig(conf['adam'])
        self.layer = LayerConfig(conf['layer'])
        self.cost_function = CostFunction(conf)
        self.num_iterations = conf['num_iterations']
        logging.info("Config init success")
        
        
if __name__ == '__main__':
    print Config().layer.activation_function
    print Config().batch_size
