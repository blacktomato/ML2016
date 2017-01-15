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
import sys
import pickle
import numpy as np
import random
from numpy import genfromtxt
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
nbEpoch = 10

#for testing
#read in the testing data
#sys.argv[1] is test file name
clickTest = genfromtxt(sys.argv[1]+'clicks_test.csv', 
                     deliniter=',', dtype='int64', skip_header=1)

#need to extract feature first
testData = [] 

#load the model
json_file = open(sys.argv[2]+'.json', 'r') #sys.argv[2] is model name
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(sys.argv[2]+'.hdf5')

#predict porbability
prob = model.predict_proba(testData, batch_size=batchSize)

#count how many times a particular display_id appear
idCount = np.bincount(clickTest[:,0])
disId = np.unique(clickTest[:,0])

start = 0
end = 0
file_dir = sys.argv[3] # sys.argv[3] is the output file name
with open(file_dir, 'w') as f:
    f.write('display_id,ad_id\n')
    for ID in disId:
        end += idCount[ID]
        #only use the clicking probability (that means class 1)
        order = np.argsort(prob[start:end,1])[::-1]
        adOrder = clickTest[order+strat] 
        start = end
        f.write(str(ID)+','+' '.join(map(str, adOrder[:,1]))+'\n')



