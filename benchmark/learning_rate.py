#encode=utf-8

from benchmark import MetaData
from benchmark import Benchmark

def test_learning_rate():
    learning_rates = [
        0.05,
        0.01,
        0.005,
        0.001,
        0.0005,
    ]
    meta_datas = [
        MetaData('learning_rate = ' + str(rate),
                 learning_rate = rate
        ) for rate in learning_rates
    ]
    Benchmark.predict_wrapper('learning_rate', meta_datas)
    
if __name__ == '__main__':
    test_learning_rate()
    