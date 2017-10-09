import l1_cost_function
import numpy as np

A = np.array([[0.3,0.1,0.2,0.2],[0.2,0.1,0.2,0.3]])
Y = np.array([[0,1,0,0],[0,0,1,0]])
A = np.ones((2,1))
Y = np.ones((2,1))
A = A.T
Y = Y.T
W = {}
W['W1'] = [1,2,3]
W['W2'] = [1,2,3]
cost = l1_cost_function.L1CostFunction.cost(A,Y,3,W)
print cost