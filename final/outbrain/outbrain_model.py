import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import tensorflow as tf
tf.python.control_flow_ops = tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)
K.get_session().run(tf.initialize_all_variables())
import sys
import pickle
import numpy as np
import random
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam, SGD, Adadelta
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization


# Constant define
batchSize = 1000
classNum = 2
nbEpoch = 30
val = 4e4
trainSize = 3.9e6

def build_model(shape):
    # Sequential
    model = Sequential()
    model.add(Dense(512, input_shape=(shape,)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(512))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(256))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(256))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(classNum))
    model.add(Activation('softmax'))
    adam = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 

#####################################################################################

filepath = 'model.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, early_stopping]

model = build_model(799)
model.fit(train_data, train_ans, batch_size=batchSize, nb_epoch=nbEpoch, 
          callbacks=callbacks_list,verbose=1, validation_data=(valid_data, valid_ans))

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
