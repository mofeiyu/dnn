#encode=utf-8

import json
import logging

logging.basicConfig(level=logging.DEBUG)

def singleton(cls):
    instance = cls()
    instance.__call__ = lambda: instance
    return instance

@singleton
class Config():
    def __init__(self, conf_path='conf.json'):
        conf = json.load(file(conf_path))
        self.learning_rate = conf['learning_rate']
        self.learning_decay = conf['learning_decay']
        self.hidden_layers = conf['hidden_layers']
        self.dropout = conf['dropout']
        self.batch_size = conf['batch_size']
        self.l1_regularization_strength = conf['l1_regularization_strength']
        self.l2_regularization_strength = conf['l2_regularization_strength']
        self.activation_function = conf['activation_function']
        self.cost_fun = conf['cost_fun']
        self.opt = conf['opt']
        self.checkpoint = conf['checkpoint']
        self.steps = conf['steps']
        logging.info("Config %s init success" % conf_path)
        
if __name__ == '__main__':
    print Config().activation_function
    print Config().batch_size

