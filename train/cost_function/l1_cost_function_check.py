import l1_cost_function
import numpy as np

A = np.array([[0.3,0.1,0.2,0.2],[0.2,0.1,0.2,0.3]])
Y = np.array([[0,1,0,0],[0,0,1,0]])
A = A.T
Y = Y.T
W = np.array([[1,2,3],[1,2,3]])
cost = l1_cost_function.L1CostFunction.l1_cost(A,Y,)
print cost