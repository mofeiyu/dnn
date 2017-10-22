# Deep Neural Network Benchmark #

We set random seed as 20171022 in order to get the same result every time.

## 1. Learning Rate ##

mini-batch

activation function: relu

optimizer: adam

 |  learning rate | epochs | run time(seconds) | train accuracy | validation accuracy | test accuracy |
 |  ------------- |:-------------: |  -----: |  -----: |  -----: |  -----: |
 | 0.05 | 30 | 61.22 | 97.93% | 95.73% | 95.46% |
 | 0.01 | 30 | 58.57 | 99.68% | 97.82% | 97.70% |
 | 0.005 | 30 | 57.42 | 100.00% | 98.17% | 98.05% |
 | 0.001 | 30 | 57.46 | 99.98% | 97.94% | 97.98% |
 | 0.0005 | 30 | 57.02 | 99.61% | 97.79% | 97.88% |

![Alt text](/img/learning_rate/cost.png)
![Alt text](/img/learning_rate/accuracy.png)

## 2. Mini-Batch ##

learning_rate: 0.001

activation function: relu

optimizer: adam

 |  mini-batch | epochs | run time(seconds) | train accuracy | validation accuracy | test accuracy |
 |  ------------- |:-------------: |  -----: |  -----: |  -----: |  -----: |
 | True | 30 | 63.48 | 99.98% | 97.94% | 97.98% |
 | False | 100 | 243.23 | 94.26% | 94.61% | 94.16% |

![Alt text](/img/mini_batch_True/cost.png)
![Alt text](/img/mini_batch_False/cost.png)
![Alt text](/img/mini_batch_True/accuracy.png)
![Alt text](/img/mini_batch_False/accuracy.png)

## 3. Activation Function ##

mini-batch

learning_rate: 0.001

optimizer: adam

 |  activation function | epochs | run time(seconds) | train accuracy | validation accuracy | test accuracy |
 |  ------------- |:-------------: |  -----: |  -----: |  -----: |  -----: |
 | sigmoid | 30 | 68.60 | 98.69% | 97.39% | 97.36% |
 | tanh | 30 | 66.02 | 99.92% | 97.74% | 97.87% |
 | relu | 30 | 59.07 | 99.98% | 97.94% | 97.98% |
 | leaky_relu | 30 | 62.67 | 91.84% | 92.48% | 91.88% |

![Alt text](/img/activation_function/cost.png)
![Alt text](/img/activation_function/accuracy.png)

## 4. Optimizer ##

mini-batch

learning_rate: 0.001

activation function: relu

 |  optimizer | epochs | run time(seconds) | train accuracy | validation accuracy | test accuracy |
 |  ------------- |:-------------: |  -----: |  -----: |  -----: |  -----: |
| sgd | 30 | 49.37 | 81.21% | 83.42% | 82.75% |
| momentum | 30 | 50.31 | 81.18% | 83.41% | 82.67% |
| rmsprop | 30 | 51.44 | 99.98% | 97.71% | 97.87% |
| adam | 30 | 57.19 | 99.98% | 97.94% | 97.98% |

![Alt text](/img/optimizer/cost.png)
![Alt text](/img/optimizer/accuracy.png)
