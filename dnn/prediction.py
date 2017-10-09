import logging

# init first
logging.basicConfig(level=logging.DEBUG) # debug log level

from dnn.test import test
from dnn.train import train
from dnn.config import Config
from dnn.load_data.load_data import MinistDataset

def main():
    config = Config()
    dataset = MinistDataset()
    parameters = train.L_layer_model(config, dataset, num_iterations = 2500, print_cost = True)
    
    #pred_test = test.test(Config, test_x, test_y, parameters)
    #print "Cost of train set is : ", pred_test

if __name__ == '__main__':
    main()
    