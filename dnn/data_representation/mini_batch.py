import numpy as np 

def mini_batch_data(X, Y, batch_size):
    mini_batchs = []
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((10,m))
    batch_num = m / batch_size + (m % batch_size > 0) 
    for j in range(batch_num):
        if j != batch_num-1:
            Xj,Yj = X[:,j*batch_size:(j+1)*batch_size],Y[:,j*batch_size:(j+1)*batch_size]
        else :
            Xj,Yj = X[:,j*batch_size:],Y[:,j*batch_size:]
        mini_batchs.append((Xj, Yj))
    return mini_batchs