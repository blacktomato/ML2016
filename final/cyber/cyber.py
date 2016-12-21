import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import tensorflow as tf
tf.python.control_flow_ops = tf
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)
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
from collections import Counter


# Constant define
batchSize = 1000
classNum = 5
nbEpoch = 10
val = 4e4
trainSize = 3.9e6
order = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

def build_model(shape):
    # Sequential
    model = Sequential()
    model.add(Dense(40, input_shape=(shape,)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))

    model.add(Dense(20))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    
    model.add(Dense(10))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))

    '''
    model.add(Dense(64))
    #model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.25))

    model.add(Dense(64))
    #model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(0.25))
    '''
    model.add(Dense(classNum))
    model.add(Activation('softmax'))
    adam = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
 


#'dictAttackBasic' is a dictionary for the basic attack types.                     
#The dictionary is used to group the sub-attack types in traning_attack_types.txt
dictAttackBasic = {'normal':0, 'dos':1, 'u2r':2, 'r2l':3, 'probe':4}
file = open(sys.argv[1]+'/training_attack_types.txt', 'r')
elementA = ['normal']
typeA = [0]
atks = file.readlines()
for atk in atks:
    atk = atk.strip().strip('.').split(' ')
    elementA.append(atk[0])
    typeA.append(dictAttackBasic[atk[1]])
dictA = dict(zip(elementA,typeA))
file.close()

#####################################################################################

#protocal, server, flag, attackType are used to store the corresponding feature in the file 'train'  
file = open(sys.argv[1]+'/train', 'r')
protocal, server, flag, attackType = [], [], [], []
server.append('icmp') #not appear in training data but in testing data
data = []
print('start parsing')
lines = file.readlines()
#data = np.zeros((len(lines),38))

for line in lines:
    #line = file.readline()
    line = line.strip().strip('.').split(',')
    protocal.append(line[1])
    server.append(line[2])
    flag.append(line[3])
    attackType.append(line[-1])
    
    del line[-1]
    #del line[12:22]
    #del line[6:11]
    del line[19]
    del line[1:4]

    data.append([int(float(a)) for a in line])
file.close()
del lines
data = np.asarray(data).astype('float32')
data = (data-np.mean(data,axis=0))/np.std(data,axis=0)
print(data[1])
#Counter is used to count the numbers of different elemnet in the list
#Counter(*).keys() return all the different element in the list 
print('create dictionary for protocal, server, flag, attackType')
pk, sk, fk, ak= Counter(protocal).keys(), Counter(server).keys(), Counter(flag).keys(), 5
pr, sr, fr = range(0,len(pk)), range(0,len(sk)), range(0, len(fk))

#Create the dictionary of protocal, server, flag
#The dictionary will be dictP = {'icmp':0, 'tcp':1, 'udp':2}
dictP, dictS, dictF = dict(zip(pk,pr)), dict(zip(sk,sr)), dict(zip(fk,fr))
del server[0] # delete 'icmp' in server list

#save the dictionary for testing(needed when using another script to do testing)
tup = (dictP, dictS, dictF)
dictionary =open('dictionary', 'wb+')
pickle.dump(tup, dictionary)
dictionary.close()
print('dictionary saved')

#use the string as an index to find the corresponding integer in the dictionary
#pl, sl, fl, al are lists store the corresponding integer for protocal, server, flag, attackType
print('string to integer')
pl, sl, fl, al = [], [], [], []
size = len(protocal)
for i in range(size):
    pl.append(dictP[protocal[i]])
    sl.append(dictS[server[i]])
    fl.append(dictF[flag[i]])
    al.append(dictA[attackType[i]])

#create identity matrix and  
print('integer to hard 1')
pi, si, fi = np.identity(len(pk)), np.identity(len(sk)), np.identity(len(fk))
protocal, server, flag = pi[pl], si[sl], fi[fl]


#concatenate protocal, server, flag
print('concatenate server/protocal')
trainData = np.concatenate((server, protocal),1)
del protocal
del server
print(trainData.shape)

print('concatenate data/flag')
trainData = np.concatenate((trainData, flag),1)
del flag

print('concatenate data/data')
#trainData = (trainData-np.mean(trainData,axis=0))/np.std(trainData,axis=0)
trainData = np.concatenate((trainData, data),1)
print(trainData.shape)
del data
#####################################################################################

#print('saving model')
#np.save('allData.npy',data)
trainData = trainData.astype('float32')
al = np.asarray(al).astype('float32')
shuffle = np.random.permutation(trainData.shape[0])
model = build_model(trainData.shape[1])

trainData, al = trainData[shuffle], al[shuffle]

trainData, al = np.split(trainData, [trainSize,trainData.shape[0]]), np.split(al, [trainSize,trainData.shape[0]])
TrainX, ValX = trainData[0], trainData[1]
TrainY, ValY = al[0], al[1]

TrainX = np.asarray(TrainX).astype('float32')
TrainY = np.asarray(TrainY).astype('int8')

ValX = np.asarray(ValX).astype('float32')
ValY = np.asarray(ValY).astype('int8')

TrainY = np_utils.to_categorical(TrainY,5)
ValY = np_utils.to_categorical(ValY,5)

filepath = sys.argv[2]+'.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint, early_stopping]

model.fit(TrainX, TrainY, batch_size=batchSize, 
          nb_epoch=nbEpoch, callbacks=callbacks_list,verbose=1, 
          validation_data=(ValX, ValY))
#model.fit(TrainX, TrainY, batch_size=batchSize, nb_epoch=nbEpoch, callbacks=callbacks_list,verbose=1)
model_json = model.to_json()
with open(sys.argv[2]+'.json', 'w') as json_file:
    json_file.write(model_json)
#####################################################################################
del TrainX
del TrainY
del ValX
del ValY

#for testing
file = open(sys.argv[1]+'/test.in', 'r')
protocal, server, flag = [], [], []
lines = file.readlines()
print('start parsing')
data2 = []
for line in lines:
    #line = file.readline()
    line = line.strip().strip('.').split(',')
    protocal.append(line[1])
    server.append(line[2])
    flag.append(line[3])
    #del line[12:22]
    #del line[6:11]
    del line[19]
    del line[1:4]

    data2.append(line)
file.close()
del lines
data2 = np.asarray(data2).astype('float32')
data2 = (data2-np.mean(data2,axis=0))/np.std(data2,axis=0)
file.close()
pr, sr, fr = range(0,len(pk)), range(0,len(sk)), range(0, len(fk))
print('string to hard 1')
pl, sl, fl, al = [], [], [], []
size = len(protocal)
for i in range(size):
    pl.append(dictP[protocal[i]])
    sl.append(dictS[server[i]])
    fl.append(dictF[flag[i]])

pi, si, fi = np.identity(len(pk)), np.identity(len(sk)), np.identity(len(fk))
protocal, server, flag = pi[pl], si[sl], fi[fl]

print('concatenate server/protocal')
testData = np.concatenate((server, protocal),1)
print('concatenate data/flag')
testData = np.concatenate((testData, flag),1)
print('concatenate data/data')
#testData = (testData-np.mean(testData,axis=0))/np.std(testData,axis=0)
testData = np.concatenate((testData, data2),1)

json_file = open(sys.argv[2]+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(sys.argv[2]+'.hdf5')

prediction = model.predict_classes(testData, batch_size=batchSize)
file_dir = sys.argv[3]
with open(file_dir, 'w') as f:
    f.write('id,label\n')
    for j in range(len(prediction)):
        f.write(str(j+1)+','+str(prediction[j])+'\n') 
