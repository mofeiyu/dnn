#encode=utf-8

from benchmark import MetaData
from benchmark import Benchmark

def test_activation_function():
    activation_functions = [
        'sigmoid',
        'tanh',
        'relu',
        'leaky_relu',
    ]
    meta_datas = [
        MetaData('activation_function = ' + str(act),
                 activation_function = act
        ) for act in activation_functions
    ]
    Benchmark.predict_wrapper('activation_function', meta_datas)
    
if __name__ == '__main__':
    test_activation_function()
    