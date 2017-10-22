import logging
from dnn.optimizers.sgd import SgdOptimizer
from dnn.optimizers.adam import AdamOptimizer
from dnn.optimizers.momentum import MomentumOptimizer
from dnn.optimizers.rmsprop import RmspropOptimizer

class OptimizerFactory():
    @staticmethod
    def get_opt(conf, layer_dims):
        opt = conf.optimizer
        logging.info(opt)
        if opt == "sgd":
            return SgdOptimizer(conf, layer_dims)
        elif opt == "adam":
            return AdamOptimizer(conf, layer_dims)
        elif opt == "momentum":
            return MomentumOptimizer(conf, layer_dims)
        elif opt == "rmsprop":
            return RmspropOptimizer(conf, layer_dims)
        else:
            return SgdOptimizer(conf, layer_dims)
        