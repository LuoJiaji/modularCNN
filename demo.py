import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


def get_random_batch(x, y, l, batchsize):
    ind_p = np.where(y_train == l)[0]
    ind_n = np.where(y_train != l)[0]
    x_batch = []
    y_batch = []
    l_p = len(ind_p)
    l_n = len(ind_n)
    for i in range(int(batchsize/2)):
        ind = random.randrange(l_p)
        x_batch.append(x[ind_p[ind]])
        y_batch.append(1)
        print(y[ind_p[ind]])
        ind = random.randrange(l_n)
        x_batch.append(x[ind_n[ind]])
        y_batch.append(0)
        print(y[ind_n[ind]])
    return x_batch, y_batch

x_batch, y_batch = get_random_batch(x_train, y_train, 0, 128)

