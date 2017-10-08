#encode=utf-8

import json
import logging

def singleton(cls):
    instance = cls()
    instance.__call__ = lambda: instance
    return instance

def get_val(conf, key, default_val):
    return conf[key] if key in conf else default_val

class AdamConfig():
    def __init__(self, conf):
        self.beta1 = get_val(conf, 'beta1', 0.9)
        self.beta2 = get_val(conf, 'beta2', 0.999)
        self.epsilon = get_val(conf, 'epsilon', 1e-8)

class RsmPropConfig():
    def __init__(self, conf):
        self.beta = get_val(conf, 'beta', 0.999)
        self.epsilon = get_val(conf, 'epsilon', 1e-8)

@singleton
class Config():
    def __init__(self, conf_path='conf.json'):
        conf = json.load(file(conf_path))
        self.data_stand = conf['data_stand']
        self.learning_rate = conf['learning_rate']
        self.learning_decay = conf['learning_decay']
        self.layers_sizes = conf['layers_sizes']
        self.initialize_parameters = conf["initialize_parameters"]
        self.dropout = conf['dropout']
        self.batch_size = conf['batch_size']
        self.layer = conf['layer']
        self.l1_regularization_strength = conf['l1_regularization_strength']
        self.l2_regularization_strength = conf['l2_regularization_strength']
        self.activation_function = conf['activation_function']
        self.cost_fun = conf['cost_fun']
        self.opt = conf['opt']
        self.checkpoint = conf['checkpoint']
        self.adam = AdamConfig(conf['adam'])
        logging.info("Config %s init success" % conf_path)
        
if __name__ == '__main__':
    print Config().activation_function
    print Config().batch_size
    