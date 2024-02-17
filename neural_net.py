# predictive anal using neural nets :)

import numpy as np


'''
    Activation for next layer of neural net
'''
def sigmoid_f(z):
    return 1 / (1 + np.e * np.exp(-z))


def cost_f():
    pass


# Cross-entropy loss
def loss_f(y, t):
    return -t * np.log2(y) - (1 - t) * np.log2(1 - y)


'''
    Compute gradient using loss func
'''
def gradient_f():
    pass


'''
    Cluster data into batches, avoid chocking ram
'''
def mini_batch(data, batch_size):
    return data


'''
    Neural Net:
        - Make sure t is in single row form
'''
def NN(X, t, epoch, lr, batch_size, n_hidden, n_output):

    batch = mini_batch(X, batch_size)

    for _ in epoch:
        pass