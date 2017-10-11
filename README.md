# Deep Nerual Network #

## 1. Introduce ##


### 1. Dataset ###

Use Mnist dataset. [Mnist Link](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz)


### 2. Normalization ###

able to use Normalization


### 3. Initialization ###

Use ramdon initialization (initialization_ramdon) or Xavier initialization (initialization)


### 4. Layer Function ###

Every layer: use linear function

Hidden layers: able to use Relu, Sigmoid, Tanh or Leaky_relu as activation function

Output layer: use Softmax


### 5. Cost Function ###

Use log-likelihood cost function (able to use L1 or L2 regularization)


### 6. Optimizers ###

Able to use Adam, SGD, SGD+Momentum, or RMSprop Optimizer

### 7. Model ###

Hyper parameter tuning : conf.json


## 2. Update ##


### 2017.10.11 13:38 ###
After 200 terators:

1.training_data (loss = 0.0316252594751, hit count=49625, hit ratio = 99.25% )

2.validation_data  (loss = 0.0316252594751, hit count=9646, hit ratio = 96.46% )

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

