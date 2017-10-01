import numpy as np
import data
import train
import test

import logging
from data import load_data
from data import normalization
from train import train
from test import test 

from config import Config

config = Config()
logging.basicConfig(level=logging.DEBUG) # debug log level

train_x, train_y, test_x, test_y = load_data.Load_Data()
if config.data_stand == "normalization":
    train_x = normalization.normalization(train_x)
parameters = train.L_layer_model(Config, train_x, train_y, num_iterations = 2500, print_cost = True)
pred_train = test.predict(Config, train_x, train_y, parameters)
pred_test = test.predict(Config, test_x, test_y, parameters)

print "Cost of train set is : ", pred_train
print "Cost of train set is : ", pred_test