import cost_function
import numpy as np
A = np.array([[0.3,0.1,0.2,0.2],[0.2,0.1,0.2,0.3]])
Y = np.array([[0,1,0,0],[0,0,1,0]])
A = A.T
Y = Y.T
cost = cost_function.CostFunction.cost(A,Y)
print cost