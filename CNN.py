# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:54:09 2019

@author: Bllue
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop, SGD
from keras.utils.vis_utils import plot_model


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_test = np.expand_dims(x_test, axis = 3)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

def get_random_batch(x, y, batchsize):
    l = len(x)
    x_batch = []
    y_batch = []
    for i in range(batchsize):
        ind = random.randrange(l)
        x_batch.append(x[ind])
        y_batch.append(y[ind])

    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    y_batch = y_batch.astype('float32')
    return x_batch, y_batch

x_batch, y_batch = get_random_batch(x_train, y_train, 128)


input_shape = (28,28,1)
input_data = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
x = Flatten(name='flatten')(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(10, activation='softmax', name='fc2')(x)

model = Model(input_data, x)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

for it in range(5000):
    x_batch, y_batch = get_random_batch(x_train, y_train, 256)
    x_batch = np.expand_dims(x_batch, axis = 3)
    train_loss, train_acc = model.train_on_batch(x_batch, y_batch)
    if it % 100 == 0:
        print('it:', it, 'loss', train_loss, 'acc', train_acc)
model.save('./models/CNN.h5')

pre = model.predict(x_test)
pre = np.argmax(pre, axis = 1)
y_test = np.argmax(y_test, axis = 1)
acc = np.mean(pre == y_test)