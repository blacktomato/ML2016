#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : cnn_fast.py
 # Purpose :
 # Creation Date : Fri 11 Nov 2016 04:50:40 PM CST
 # Last Modified : Sat 12 Nov 2016 05:44:44 PM CST
 # Created By : SL Chung
##############################################################
import numpy as np
import pickle
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam


label = np.load("./Label.npy")      #10-500-3072
unlabel = np.load("./Unlabel.npy")  #45000-3072

label = label.reshape((5000, 3072))
xtrain = label.reshape((5000, 3, 32, 32))
ytrain = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]*500)
ytrain = np.transpose(ytrain.reshape((500,10,10)), (2,0,1)).reshape(5000,10)

#Start training
#Convolution
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(50,3,3))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
#Fully connected
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

#Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain, ytrain, batch_size=200, nb_epoch=1)

#Validation
result = model.evaluate(xtrain,ytrain,batch_size=10000)
print(result[1])
