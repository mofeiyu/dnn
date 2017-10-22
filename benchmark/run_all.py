#encode=utf-8

from best_choice import best_choice
from optimizer import test_optimizer
from mini_batch import test_mini_batch
from learning_rate import test_learning_rate
from activation_function import test_activation_function

def main():
    print 'test_optimizer'
    test_optimizer()
    print 'test_mini_batch'
    test_mini_batch()
    print 'test_learning_rate'
    test_learning_rate()
    print 'test_activation_function'
    test_activation_function()
    print 'best_choice'
    best_choice()
    
if __name__ == '__main__':
    main()
    