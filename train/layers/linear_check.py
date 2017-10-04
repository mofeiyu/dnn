import numpy as np
import linear
from linear import LinearLayer

A = np.array([[0.3,0.1,0.2,0.2],[0.2,0.1,0.2,0.3]]).T
print A.shape
W = np.array([[0.3,0.1,0.2,0.2]])
print W.shape
b = np.array([[1]])


print LinearLayer.linear_forward(A, W, b)