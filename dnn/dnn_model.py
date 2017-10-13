#encode=utf-8

import numpy as np

from dnn.config import Config
from dnn.load_data.prepare_data import ReaderWrapper
from dnn.data_representation.mini_batch import mini_batch_data
from dnn.data_representation.normalization import normalization
from dnn.cost_function.test_accuracy import test_accuracy
from dnn.cost_function.train_accuracy import train_accuracy

from dnn.initialization.initialization_factory import InitializationFactory
from dnn.layers.layer_factory import LayerFactory
from dnn.cost_function.cost_function_factory import CostFunctionFactory
from dnn.optimizers.optimizer_factory import OptimizerFactory

class DnnModel:
    def __init__(self):
        self._reader = ReaderWrapper()
        self._config = Config()
        
    def _build_model(self):
        self._initialization = InitializationFactory.get_initialization(self._config.initialization)
        self._cost_function = CostFunctionFactory.get_cost_function(self._config.cost_function.cost_function)
        self._hidden_layer, self._output_layer = LayerFactory.get_layer(self._config.layer.activation_function)
        self._opt = OptimizerFactory.get_opt(self._config, self._layer_dims)           
        
    def train(self):
        train_X, train_Y = self._reader.get_training_data()
        self._layer_dims = self._config.layer.hidden_units
        self._layer_dims.insert(0,train_X.shape[0])
        self._layer_dims.append(self._config.layer.n_classes)
        self._build_model()
        self._parameters = self._initialization(self._layer_dims)
        self._L = len(self._layer_dims) - 1
        self._costs = []
        self._grads = {}
        self._caches = []       
        self._AL = None
        self._mean = None
        self._var = None
        if self._config.normalization:
            _, self._mean, self._var = normalization(train_X)
        basic_l_r = self._config.learning_rate
        for i in range(self._config.num_iterations):            
            self._config.learning_rate = 1.0/(1 + self._config.learning_decay * i) * basic_l_r
            if self._config.mini_batch.mini_batch:
                mini_batchs = mini_batch_data(train_X, train_Y, self._config.mini_batch.batch_size)
            hit_count = 0
            for X, Y in mini_batchs:                
                self._forward(X)          
                cost = self._cost_function.cost(self._AL, Y, self._L, self._parameters)         
                self._backward(Y)
                self._parameters = self._opt.update_parameters(self._parameters, self._grads, self._L)
                hit_count += train_accuracy(self._AL, Y)
            print ('step %d cost %f, hit_count = %d, hit_ratio = %.2lf%%' % (i, cost, hit_count, hit_count * 100.0 / train_X.shape[1]))

    
    def _predict(self, A, Y):
        if self._config.normalization:
            A = np.divide(A - self._mean, self._var)      
        for l in range(1, self._L):
            A,_  = self._hidden_layer.forward(A, self._parameters['W' + str(l)], self._parameters['b' + str(l)])
        AL,_ = self._output_layer.forward(A,self._parameters['W' + str(self._L)], self._parameters['b' + str(self._L)])
        cost = self._cost_function.cost(AL, Y, self._L, self._parameters)
        hit_count = test_accuracy(AL, Y)
        print ('cost %f, hit_count = %d, hit_ratio = %.2lf%%' % (cost, hit_count, hit_count * 100.0 / A.shape[1]))
    
    def test(self):
        X, Y = self._reader.get_test_data()
        self._predict(X, Y)
        
    def validation(self):
        X, Y = self._reader.get_validation_data()
        self._predict(X, Y)

    def _forward(self, X):
        caches = []
        A = X
        if self._config.normalization:
            A, _, _ = normalization(A)
        for l in range(1, self._L):
            A, cache = self._hidden_layer.forward(A, self._parameters['W' + str(l)], self._parameters['b' + str(l)])
            caches.append(cache)
        AL, cache = self._output_layer.forward(A,self._parameters['W' + str(self._L)], self._parameters['b' + str(self._L)])  
        caches.append(cache)
        self._caches = caches
        self._AL = AL

    def _backward(self, Y):
        grads = {}
        current_cache = self._caches[-1]
        dZ = self._AL - Y
        
        dA_prev_temp, dW_temp, db_temp = self._output_layer.backward(dZ, current_cache, self._config.cost_function.cost_function, self._config.cost_function.regularization_strength)    
        grads["dA" + str(self._L)] = dA_prev_temp
        grads["dW" + str(self._L)] = dW_temp
        grads["db" + str(self._L)] = db_temp
        
        for l in reversed(range(self._L-1)):
            current_cache = self._caches[l]
            dA_prev_temp, dW_temp, db_temp = self._hidden_layer.backward(grads["dA" + str(l + 2)], current_cache, self._config.cost_function.cost_function, self._config.cost_function.regularization_strength)
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        self._grads = grads
