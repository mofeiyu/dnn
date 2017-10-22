import numpy as np

def normalization(X):
    m = X.shape[1]
    u = np.sum(X, axis = 1, keepdims = True) * 1.0 / m
    X = X - u
    v = np.sum(X**2, axis = 1, keepdims = True) * 1.0 / m
    v = v**0.5 + 1e-8
    X_norm = np.divide(X, v)
    return X_norm, u, v
