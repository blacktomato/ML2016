#usr/bin/env python3
# coding=utf-8
##############################################################
 # Purpose :
 # Creation Date : Fri 11 Nov 2016 04:50:40 PM CST
 # Last Modified : Fri 18 Nov 2016 17:57:01 CST
 # Created By : SL Chung
##############################################################
import numpy as np
import pickle
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, UpSampling2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model, load_model

#load the model
model = load_model(sys.argv[1])
encoder = load_model(sys.argv[2])


test = np.load("./Test.npy")        #10000-3072
xtest = test.reshape((10000, 3, 32, 32)) / 255.

xtest = encoder.predict(xtest)
test_result = model.predict(xtest, batch_size=100, verbose=0)

total = np.array([0]*10)

#output file
output = open(sys.argv[3], "w+")
output.write("ID,class\n")

for i in range(len(test_result)):
    total[int(np.argmax(test_result[i]))] += 1
    line = str(i) + "," + str(int(np.argmax(test_result[i]))) + "\n"
    output.write(line)
output.close()

print(total)
print("Output file:", sys.argv[3]) 

