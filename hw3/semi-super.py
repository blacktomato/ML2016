#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : semi-super_fast.py
 # Purpose : Use the data process by numpy and do the semi-supervise learing 
 # Creation Date : Fri 11 Nov 2016 04:50:40 PM CST
 # Last Modified : Sat 19 Nov 2016 00:14:19 CST
 # Created By : SL Chung
##############################################################
import numpy as np
import pickle
import sys

from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

model = Sequential()
#Convolution
model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(3, 32, 32)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Convolution2D(50, 3, 3))
model.add(Activation('relu'))

model.add(Convolution2D(80, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero',
                   gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
#Fully connected
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es=EarlyStopping(monitor='loss', min_delta=0.00001, patience=5, verbose=0, mode='auto')

def train_cifar10(X, Y, epoch, batch, datagen):
    #Training
    datagen.fit(X)
    model.fit_generator(datagen.flow(X, Y, batch_size=batch), callbacks=[es]
                        ,samples_per_epoch=len(X), nb_epoch=epoch)
    #model.fit(xtrain[train], ytrain[train], batch_size=300, nb_epoch=100)
    return model

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

#shuffle for validation
xtrain, ytrain = shuffle(xtrain, ytrain, random_state = 0)
#fix random seed for reproducibility

datagen = ImageDataGenerator(
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.02,   # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.02)  # randomly shift images vertically (fraction of total height)

#Start training
each = 500
while(len(unlabel) > 8000):
    epoch = 80
    batch = 300
    model = train_cifar10(xtrain, ytrain, epoch, batch, datagen)
    if (len(unlabel)==0):
        break
    else:
        unlabel_result = model.predict(unlabel, batch_size=100, verbose=0)

        temp_max = unlabel_result.max(axis=1)
        #get the confident data(maybe the right answer)
        confident_data = ((temp_max > (0.8) )*1).nonzero()[0]
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
            adding_index_max = np.hstack(( np.random.choice((class_max == i).nonzero()[0], increase, replace=True),
                               adding_index_max))
        adding_index_max = adding_index_max.astype(int)
        
        #remove some of the confident_data the unlabel 
        training_unlabel = unlabel[confident_data[adding_index_max]]
        unlabel = np.delete(unlabel, confident_data[adding_index_max], axis=0)
        testing_unlabel = np.identity(10, dtype=int)[class_max[adding_index_max]]

        xtrain = np.vstack(( xtrain, training_unlabel ))
        ytrain = np.vstack(( ytrain, testing_unlabel  ))
        print(np.sum(ytrain, axis=0))

    print(len(xtrain))

epoch = 40
batch = 500
model = train_cifar10(xtrain, ytrain, epoch, batch, datagen)

model.save(sys.argv[1])
