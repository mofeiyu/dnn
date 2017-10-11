

import numpy as np

Z = np.array([[0,1,0,0],[0,0,1,0]])
print Z.shape

S = np.exp(Z)/np.sum(np.exp(Z),axis = 0, keepdims = True)
print S