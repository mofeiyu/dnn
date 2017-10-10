# Deep Nerual Network #

## 1. Introduce ##

## 2. Update ##

Use Mnist dataset. [Mnist Link](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz)

### 2017.10.11 13:38 ###
After 200 terators:

1.training_data (loss = 0.0316252594751, hit count=49625, hit ratio = 99.25% )
2.validation_data  (loss = 0.0316252594751, hit count=9646, hit ratio = 96.46% )


### 2017.10.10 00:12 ###

After 450 iterators:

1. loss = 0.280660009558

2. hit count=45971

3. hit ratio = 91.94%

~~~ json
{
    "learning_rate": 0.05,
    "layers_sizes": [784, 128, 64, 10],
    "layer":"relu",
    "activation_function": "relu",
    "cost_fun": "lr_cost_function",
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
    "layer":"relu",
    "activation_function": "relu",
    "cost_fun": "lr_cost_function",
    "opt": "sgd",
    "initialize_parameters":"initialization_he"
}
~~~

