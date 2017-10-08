import logging

from dnn.test import test
from dnn.train import train
from dnn.config import Config
from dnn.load_data.load_data import MinistDataset

def main():
    config = Config()
    logging.basicConfig(level=logging.DEBUG) # debug log level
    
    dataset = MinistDataset()
    
    # (img_list, label_list)
    #print dataset.get_training_data()[1].shape
    
    parameters = train.L_layer_model(Config, dataset, num_iterations = 2500, print_cost = True)
    
    #pred_test = test.test(Config, test_x, test_y, parameters)
    #print "Cost of train set is : ", pred_test

if __name__ == '__main__':
    main()
    