import numpy as np

def normalition(X):
    m = X.shape[1]
    u = np.sum(X, axis = 1, keepdims = True)/m
    X = X - u
    v2 = np.sum( X**2, axis = 1, keepdims = True)/m
    X_norm = X/v2 
    return X_norm