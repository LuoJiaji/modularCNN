import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop, SGD


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
#        print(y[ind_p[ind]])
        ind = random.randrange(l_n)
        x_batch.append(x[ind_n[ind]])
        y_batch.append(0)
#        print(y[ind_n[ind]])
    return x_batch, y_batch

x_batch, y_batch = get_random_batch(x_train, y_train, 0, 128)

input_shape = (28,28,1)
input_data = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
x = Flatten(name='flatten')(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(1, activation='relu', name='fc2')(x)
model = Model(input_data, x)


