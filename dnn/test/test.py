from dnn.train import get_
from dnn.train.layers import softmax
import test_cost

def test(config,dataset, parameters):
    X, Y = dataset.validation_data.X, dataset.validation_data.Y
    A = X
    act_layer, cost_function, opt= get_.get(config)
    L = len(parameters) // 2  
    for l in range(1, L):
        A, cache = act_layer.forward(A, parameters['W' + str(l)], parameters['b' + str(l)])
    AL, cache = softmax.softmax.forward(A,parameters['W' + str(L)], parameters['b' + str(L)])  
    print test_cost.calculate_hit_count(AL, Y)
    return test_cost.calculate_hit_count(AL, Y)