# Deep Neural Network #

## 1. Table of contents ##

* [Features](#features)
* [Supported networks](#supported-networks)
* [Examples](#examples)


## 2. Features ##

- Reasonably fast, without GPU:

    - 100% accuracy on training dataset(Mnist dataset[Mnist Link](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz)) in about 3 minutes training, and 98% in validation dataset (adam + relu + softmax + mini_bacth)

- Singleton Pattern:

    - Safe , easy to tune hyper parameter and build the network as you need

- Simple factory(static factory method):

    - A good way to maintain and new class

    - Modify your code with less change

## 3. Supported networks ##

### layer-types ###

- core

    - fully-connected

    - linear operation

- normalization

    - normalization 

    - mini batch normalization 

### activation functions ###

* tanh

* sigmoid

* softmax

* rectified linear(relu)

* leaky relu

### loss function ###

* log-likelihood (with/without L1/L2 regularization)


### optimization algorithms ###

* stochastic gradient descent(sgd) 

* momentum

* rmsprop

* adam

(with/without L1/L2 regularization)

## 4. Benchmark ##

[Benchmark](/benchmark.md)

## 5. Examples ##

### 2017.10.12 10.17 ###

After 30 iterators:

1. training_data (hit ratio = 100% )

2. validation_data (hit ratio = 98.12% )

3. test_data (hit ratio = 98.15%)

~~~ json
{   
    "learning_rate": 0.001,
    "batch_size": 512,
    "cost_function": "log-likelihood",
    "num_iterations": 30,
    "optimizer": "adam",
    "initialization": "initialization_Xavier",
    "normalization" : false, 
    "adam": {
        "beta1" : 0.9,
        "beta2" : 0.999,
        "epsilon" : 1e-8
    },
    "layer": {
        "layers_sizes": [784, 512, 10],
        "activation_function": "relu",
        "output_layer": "softmax"
    }
}
~~~


### 2017.10.11 13:38 ###
After 200 terators:

1. training_data (loss = 0.0316252594751, hit count=49625, hit ratio = 99.25% )

2. validation_data  (loss = 0.0316252594751, hit count=9646, hit ratio = 96.46% )

~~~ json
{
    "learning_rate": 0.05,
    "layers_sizes": [784, 128, 10],
    "activation_function": "relu",
    "cost_fun": "cost_function",
    "opt": "adam",
    "initialize_parameters":"initialization_he",
    "adam": {
        "beta1": 0.9,
        "beta2": 0.999
    }
}
~~~


### 2017.10.10 00:12 ###

After 450 iterators:

1. loss = 0.280660009558

2. hit count=45971

3. hit ratio = 91.94%

~~~ json
{
    "learning_rate": 0.05,
    "layers_sizes": [784, 128, 64, 10],
    "activation_function": "relu",
    "cost_fun": "cost_function",
    "opt": "adam",
    "initialize_parameters":"initialization_he",
    "adam": {
        "beta1": 0.9,
        "beta2": 0.999
    }
}
~~~

### 2017.10.09 23:57 ###

After 450 iterators:

1. loss = 0.280660009558

2. hit count=45971

3. hit ratio = 91.94%

~~~ json
{
    "learning_rate": 0.05,
    "layers_sizes": [784, 128, 64, 10],
    "activation_function": "relu",
    "cost_fun": "cost_function",
    "opt": "sgd",
    "initialize_parameters":"initialization_he"
}
~~~

