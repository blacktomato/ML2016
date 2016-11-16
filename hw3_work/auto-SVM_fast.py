#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : auto-SVM_fast.py
 # Purpose : Use the data process by numpy and do the autoencoder-SVM learing 
 # Creation Date : Fri 11 Nov 2016 04:50:40 PM CST
 # Last Modified : Thu 17 Nov 2016 02:45:49 CST
 # Created By : SL Chung
##############################################################
import numpy as np
import pickle
import sys

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model

input_img = Input(shape=(3, 32, 32))

x = Convolution2D(30, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(10, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(10, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(30, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
es=EarlyStopping(monitor='loss', min_delta=0.00001, patience=5, verbose=0, mode='auto')

def train_cifar10(X, Y, epoch, batch, datagen):
    #Training
    autoencoder.fit_generator(datagen.flow(X, Y, batch_size=batch), callbacks=[es]
                            ,samples_per_epoch=len(X), nb_epoch=epoch)
    #model.fit(xtrain[train], ytrain[train], batch_size=300, nb_epoch=100)
    return autoencoder

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

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

datagen = ImageDataGenerator(
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.02,   # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.02)  # randomly shift images vertically (fraction of total height)

#Start training
while(len(unlabel) > 8000):
    epoch = 60
    batch = 300
    train_cifar10(xtrain, xtrain, epoch, batch, datagen):
    if (len(unlabel)==0):
        break
    else:
        unlabel_result = model.predict(unlabel, batch_size=100, verbose=0)

        temp_max = unlabel_result.max(axis=1)
        class_max = unlabel_result.argmax(axis=1)
        confident_data = (temp_max > (unlabel_result.sum(axis=1) * 0.8) )*1
        adding_index_max = confident_data.nonzero()
        training_unlabel = unlabel[adding_index_max]
        unlabel = np.delete(unlabel, adding_index_max, axis=0)
        testing_unlabel = np.identity(10, dtype=int)[class_max[adding_index_max]]

        xtrain = np.vstack(( xtrain, training_unlabel ))
        ytrain = np.vstack(( ytrain, testing_unlabel  ))
        print(np.sum(ytrain, axis=0))
    print(len(xtrain))




#test_result





total = np.array([0]*10)
#output file
output = open(sys.argv[1], "w+")
output.write("ID,class\n")

for i in range(len(test_result)):
    total[int(np.argmax(test_result[i]))] += 1
    line = str(i) + "," + str(int(np.argmax(test_result[i]))) + "\n"
    output.write(line)
output.close()

print(total)
print("Output file:", sys.argv[1]) 

