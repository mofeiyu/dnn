#encode=utf-8

from benchmark import MetaData
from benchmark import Benchmark

def best_choice():
    meta_datas = [
        MetaData('best-choice'),
    ]
    Benchmark.predict_wrapper('best_choice', meta_datas)
    
if __name__ == '__main__':
    best_choice()
    