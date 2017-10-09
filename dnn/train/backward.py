from dnn.train.layers.softmax import softmax


def backward(config, AL, Yj, act_layer, caches):
    grads = {}
    L = len(caches)
    current_cache = caches[-1]
    dZ = AL - Yj

    dA_prev_temp, dW_temp, db_temp = softmax.backward(dZ, current_cache)

    grads["dA" + str(L)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = act_layer.backward(grads["dA" + str(l + 2)], current_cache)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads
