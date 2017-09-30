#encode=utf-8

import json

class Config:
    def __init__(self, conf_path='conf.json'):
        conf = json.load(file(conf_path))
        self.data = conf['data']
        self.learning_rate = conf['learning_rate']
        self.learning_decay = conf['learning_decay']
        self.layers_sizes = conf['layers_sizes']
        self.dropout = conf['dropout']
        self.batch_size = conf['batch_size']
        self.l1_regularization_strength = conf['l1_regularization_strength']
        self.l2_regularization_strength = conf['l2_regularization_strength']
        self.activation_function = conf['activation_function']
        self.cost_fun = conf['cost_fun']
        self.opt = conf['opt']
        self.checkpoint = conf['checkpoint']
    