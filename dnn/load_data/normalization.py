import numpy as np

def normalization(X):
    print "normalization"
    m = X.shape[1]
    u = np.sum(X, axis = 1, keepdims = True) * 1. / m
    X = X - u
    v = np.sum(X**2, axis = 1, keepdims = True) * 1. / m
    X_norm = np.divide(X, v)
    return X_norm, u, v
