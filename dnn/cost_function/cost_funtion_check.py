import cost_function
import numpy as np
A = np.array([[np.exp(2),0.1,np.exp(3),np.exp(1)],[np.exp(2),np.exp(1),0.1,np.exp(1)],[1,1,2,0.1]])
Y = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1]])
A = A.T
Y = Y.T

print np.log(1)
print np.multiply(np.log(A), Y)
cost = cost_function.CostFunction.cost(A,Y)
print cost