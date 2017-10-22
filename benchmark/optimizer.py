#encode=utf-8

from benchmark import MetaData
from benchmark import Benchmark

def test_optimizer():
    optimizers = [
        'sgd',
        'momentum',
        'rmsprop',
        'adam',
    ]
    meta_datas = [
        MetaData('optimizer = ' + str(opt),
                 optimizer = opt
        ) for opt in optimizers
    ]
    Benchmark.predict_wrapper('optimizer', meta_datas)
    
if __name__ == '__main__':
    test_optimizer()
    