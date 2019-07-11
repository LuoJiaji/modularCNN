import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop, SGD


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

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
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    y_batch = y_batch.astype('float32')
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
x = Dense(1, activation='sigmoid', name='fc2')(x)
model = Model(input_data, x)


#model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

for i in range(10):
    for it in range(5000):
        x_batch, y_batch = get_random_batch(x_train, y_train, i, 256)
        x_batch = np.expand_dims(x_batch, axis = 3)
        train_loss, train_acc = model.train_on_batch(x_batch, y_batch)
        if it % 100 == 0:
            print('i:', i, 'it:', it, 'loss', train_loss, 'acc', train_acc)
    model.save('./models/ModularCNN_' + str(i) + '.h5')

#model = load_model('./models/ModularCNN_0.h5')
#test_label = np.copy(y_test)
#test_label[np.where(y_test == 0)] = 1
#test_label[np.where(y_test != 0)] = 0  
#x_test = np.expand_dims(x_test, axis = 3)
#pre = model.predict(x_test)
#pre = pre[:,0]
#pre[np.where(pre < 0.2)] = 0
#pre[np.where(pre >= 0.2)] = 1
#
#acc = np.mean(pre == test_label)