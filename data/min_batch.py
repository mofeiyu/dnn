import normalization
def mini_batch_data(X,Y,j,batch_size,batch_num,data_stand):
    if j != batch_num-1:
        Xj,Yj = X[:,j*batch_size:(j+1)*batch_size],Y[:,j*batch_size:(j+1)*batch_size]
        if data_stand == "batch_normalization":
            Xj, = normalization.normalization(Xj)
    else :
        Xj,Yj = X[:,j*batch_size:],Y[:,j*batch_size:]
        if data_stand == "batch_normalization":
            Xj, = normalization.normalization(Xj) 
    return Xj,Yj  