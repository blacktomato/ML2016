#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : cnn_fast.py
 # Purpose :
 # Creation Date : Fri 11 Nov 2016 04:50:40 PM CST
 # Last Modified : Mon 14 Nov 2016 11:52:14 PM CST
 # Created By : SL Chung
##############################################################
import numpy as np
import pickle
import sys

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

def train_cifar10(X, Y, epoch, batch, datagen):
    model = Sequential()
    #Convolution
    model.add(Convolution2D(30, 5, 5, border_mode='same', input_shape=(3, 32, 32)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(60, 4, 4))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    #Fully connected
    model.add(Dense(100))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()

    #Training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    datagen.fit(X)
    model.fit(datagen.flow(X, Y, batch_size=batch),validation_split=0.1, nb_epoch=epoch)
    #model.fit(xtrain[train], ytrain[train], batch_size=300, nb_epoch=100)
    #scores = model.evaluate(xtrain[test], ytrain[test], verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #cvscores.append(scores[1] * 100)

    #Validation
    #print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    return model

label = np.load("./Label.npy")      #10-500-3072
unlabel = np.load("./Unlabel.npy")  #45000-3072
test = np.load("./Test.npy")        #10000-3072

label = label.reshape((5000, 3072))
unlabel = unlabel.reshape((45000, 3, 32, 32)) / 255.
xtrain = label.reshape((5000, 3, 32, 32))/ 255.
xtest = test.reshape((10000, 3, 32, 32)) / 255.
temp = np.array([np.identity(10, dtype=int)]*500)
ytrain = np.transpose(temp, (2,0,1)).reshape(5000,10)

#shuffle for validation
xtrain, ytrain = shuffle(xtrain, ytrain, random_state = 0)
#fix random seed for reproducibility
#seed = 7
#np.random.seed(seed)
#define 10-fold cross validation test harness
#kfold = KFold(n_splits=10)
#cvscores=[]

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

#Start training
epoch = 100
batch = int(len(xtrain) / 10)
model = train_cifar10(xtrain, ytrain, epoch, batch, datagen)
#for train, test in kfold.split(xtrain, ytrain):

unlabel_result = model.predict(unlabel, batch_size=32, verbose=0)

temp_max = unlabel_result.max(axis=1)
class_max = unlabel_result.argmax(axis=1)
index_max = (temp_max > (unlabel_result.sum(axis=1) / 2) ).nonzero()
training_unlabel = unlabel[index_max]
testing_unlabel = np.identity(10, dtype=int)[class_max[index_max]]

xtrain = np.vstack(( xtrain, training_unlabel ))
ytrain = np.vstack(( ytrain, testing_unlabel  ))

#Training again
epoch = 100
batch = int(len(xtrain) / 10)
model = train_cifar10(xtrain, ytrain, epoch, batch, datagen)


test_result = model.predict(xtest, batch_size=100, verbose=0)
#output file
output = open(sys.argv[1], "w+")
output.write("ID,class\n")

for i in range(len(test_result)):
print(test_result[i])
line = str(i) + "," + str(int(np.argmax(test_result[i]))) + "\n"
output.close()

print("Output file:", sys.argv[1]) 

