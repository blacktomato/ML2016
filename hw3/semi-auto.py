#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : semi_auto.py
 # Purpose : Use the data process by numpy and do the autoencoder-SVM learing 
 # Creation Date : Fri 11 Nov 2016 04:50:40 PM CST
 # Last Modified : Sat 19 Nov 2016 00:59:18 CST
 # Created By : SL Chung
##############################################################
import numpy as np
import pickle
import sys

from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, UpSampling2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model

model = Sequential()
#Convolution
model.add(Convolution2D(10, 2, 2, border_mode='same', input_shape=(8, 4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Convolution2D(30, 2, 2))
model.add(Activation('relu'))
model.add(Dropout(0.1))


model.add(Flatten())
model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                   gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
#Fully connected
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es=EarlyStopping(monitor='loss', min_delta=0.00001, patience=5, verbose=0, mode='auto')

def train_cifar10(X, Y, epoch, batch):
    #Training
    model.fit(X, Y, batch_size=batch, callbacks=[es], nb_epoch=epoch)
    #model.fit(xtrain[train], ytrain[train], batch_size=300, nb_epoch=100)
    return model

input_img = Input(shape=(3, 32, 32))

x = Convolution2D(30, 5, 5, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(15, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(15, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)
# at this point the representation is (8, 4, 4) i.e. 128-dimensional


x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(15, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(15, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(30, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='relu', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

label = np.load("./Label.npy")      #10-500-3072
unlabel = np.load("./Unlabel.npy")  #45000-3072
test = np.load("./Test.npy")        #10000-3072

label = label.reshape((5000, 3072))
unlabel = unlabel.reshape((45000, 3, 32, 32)) / 255.
xtrain = label.reshape((5000, 3, 32, 32))/ 255.
xtest = test.reshape((10000, 3, 32, 32)) / 255.
unlabel = np.vstack((unlabel, xtest))
temp = np.array([np.identity(10, dtype=int)]*500)
ytrain = np.transpose(temp, (2,0,1)).reshape(5000,10)

epoch = 50
batch = 300
autoencoder.fit(xtrain, xtrain, batch_size=batch, nb_epoch=epoch)

unlabel = encoder.predict(unlabel)
xtrain = encoder.predict(xtrain)

#shuffle for validation
xtrain, ytrain = shuffle(xtrain, ytrain, random_state = 0)

#Start training
each = 500
while(len(unlabel) > 8000):
    epoch = 200
    batch = 300
    model = train_cifar10(xtrain, ytrain, epoch, batch)
    if (len(unlabel)==0):
        break
    else:
        unlabel_result = model.predict(unlabel, batch_size=100, verbose=0)

        temp_max = unlabel_result.max(axis=1)
        #get the confident data(maybe the right answer)
        confident_data = ((temp_max > 0.8)*1).nonzero()[0]
        #only get the answer for confident data
        class_max = unlabel_result.argmax(axis=1)[confident_data]
        
        #Adding same number of data for each class
        counting = np.bincount(class_max)
        increase = np.min(counting) - each
        if (increase <= 0):
            break
        each = np.min(counting)

        #index for confident_data
        adding_index_max = np.array(())
        for i in range(10):
            adding_index_max = np.hstack(( np.random.choice((class_max == i).nonzero()[0],
                                increase, replace=True), adding_index_max))
        adding_index_max = adding_index_max.astype(int)
        
        #remove some of the confident_data the unlabel 
        training_unlabel = unlabel[confident_data[adding_index_max]]
        unlabel = np.delete(unlabel, confident_data[adding_index_max], axis=0)
        testing_unlabel = np.identity(10, dtype=int)[class_max[adding_index_max]]

        xtrain = np.vstack(( xtrain, training_unlabel ))
        ytrain = np.vstack(( ytrain, testing_unlabel  ))
        print(np.sum(ytrain, axis=0))

    print(len(xtrain))

epoch = 50
batch = 500
model = train_cifar10(xtrain, ytrain, epoch, batch)

model.save(sys.argv[1])
encoder.save(sys.argv[2])
