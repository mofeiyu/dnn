
import gzip
import cPickle
import numpy as np

class Dataset:
    def __init__(self, X, Y):
        self.X = X.T
        self.Y = Y.T
        
    def data_size(self):
        return len(self.X)
    
class MinistDataset:
    def __init__(self, data_path = '../data/mnist.pkl.gz'):
        self._input_shape = (784)
        self._result_shape = (10)
        self._data_path = data_path
        self._load_data_wrapper()
        
    def _load_data(self):
        f = gzip.open(self._data_path, 'rb')
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
        self.training_data = Dataset(np.array(training_inputs), np.array(training_results))
        validation_inputs = [np.reshape(x, self._input_shape) for x in va_d[0]]
        self.validation_data = Dataset(np.array(validation_inputs), np.array(va_d[1]))
        test_inputs = [np.reshape(x, self._input_shape) for x in te_d[0]]
        self.test_data = Dataset(np.array(test_inputs), np.array(te_d[1]))

if __name__ == '__main__':
    reader = MinistDataset('../../data/mnist.pkl.gz')
    print list(reader.training_data.X)
