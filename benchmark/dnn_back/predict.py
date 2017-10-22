#encode=utf-8

import logging
# init first
logging.basicConfig(level=logging.INFO) # debug log level

from dnn.dnn_model import DnnModel

def main():
    dnn_model = DnnModel()
    dnn_model.train()
    dnn_model.validation()
    dnn_model.test()
    
if __name__ == '__main__':
    main()
    