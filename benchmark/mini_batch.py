#encode=utf-8

from benchmark import MetaData
from benchmark import Benchmark

def test_single(mini_batch, num_iterations, sample):
    meta_datas = [
        MetaData(
            'mini_batch = ' + str(mini_batch),
             mini_batch = mini_batch,
             num_iterations = num_iterations,
             sample = sample,
             )
    ]
    Benchmark.predict_wrapper('mini_batch_' + str(mini_batch), meta_datas)    

def test_mini_batch():
    test_single(True, 30, 30)
    test_single(False, 100, 1)
    
if __name__ == '__main__':
    test_mini_batch()
    