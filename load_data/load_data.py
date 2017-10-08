
import gzip
import cPickle
import numpy as np

class MinistDataset:
    def __init__(self):
        self._input_shape = (784)
        self._result_shape = (10)
        self._load_data_wrapper()
        
    def _load_data(self):
        f = gzip.open('../data/mnist.pkl.gz', 'rb')
        training_data, validation_data, test_data = cPickle.load(f)
        f.close()
        return (training_data, validation_data, test_data)
    
    def _vectorized_result(self, j):
        e = np.zeros(self._result_shape)
        e[j] = 1
        return e
    
    def _load_data_wrapper(self):
        tr_d, va_d, te_d = self._load_data()
        training_inputs = [np.reshape(x, self._input_shape) for x in tr_d[0]]
        training_results = [self._vectorized_result(y) for y in tr_d[1]]
        self._training_data = (np.array(training_inputs).T,
                               np.array(training_results).T)
        
        validation_inputs = [np.reshape(x, self._input_shape) for x in va_d[0]]
        self._validation_data = (np.array(validation_inputs).T,
                                 np.array(va_d[1]).T)
        
        test_inputs = [np.reshape(x, self._input_shape) for x in te_d[0]]
        self._test_data = (np.array(test_inputs).T,
                           np.array(te_d[1]).T)
        
    def get_training_data(self):
        return self._training_data
    
    def get_validation_data(self):
        return self._validation_data
    
    def get_test_data(self):
        return self._test_data

if __name__ == '__main__':
    pass
