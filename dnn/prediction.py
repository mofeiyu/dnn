import logging

# init first
logging.basicConfig(level=logging.DEBUG) # debug log level

from dnn.test import test, validation
from dnn.train import train
from dnn.config import Config
from dnn.load_data.load_data import MinistDataset

def main():
    config = Config()
    dataset = MinistDataset()
    parameters = train.L_layer_model(config, dataset, num_iterations = 400, print_cost = True)
    pred_validation = validation.validation(config, dataset, parameters)    
    pred_test = test.test(config, dataset, parameters)
    #print "Cost of train set is : ", pred_test

if __name__ == '__main__':
    main()
    