#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : data_processing.py
 # Purpose : preprocess the data for validtion
 # Creation Date : Thu 27 Oct 2016 10:48:19 AM CST
 # Last Modified : Thu 27 Oct 2016 11:03:31 AM CST
 # Created By : SL Chung
##############################################################
import math
import numpy as np
import sys
import random
import pickle

train_file = open(sys.argv[1], "r", encoding='utf-8', errors='ignore')
train_data = train_file.read().splitlines()

data = np.array(())
answer = np.array(())
for i in range(len(train_data)):
    #remove id for each data
    data_element = train_data[i].split(',')[1::]
    data_temp = np.array(())
    #data processing
    for j in range(57):
        data_temp = np.hstack(( data_temp, np.array( float(data_element[j]) ) ))
    #answer
    if (i == 0):
        data = np.hstack(( data, data_temp ))
    else:
        data = np.vstack(( data, data_temp ))
    answer = np.hstack(( answer, np.array( float(data_element[57]) ) ))

#Normalization
mean = np.sum(data, axis=0)/len(train_data)
std_s = (np.sum((data - mean) ** 2, axis=0)/len(train_data) ) ** 0.5

data = (data - mean) / std_s

trainings = []
validations = []
t_ans = []
v_ans = []

for i in range(10):
    validations.append(data[0:400])
    v_ans.append(answer[0:400])
    trainings.append(data[400::])
    t_ans.append(answer[400::])
    data = np.roll(data, 400, axis=0)
    answer = np.roll(answer, 400)

#output the preprocessed the data
valid_data = open(sys.argv[2], "wb+")
bind = (validations, v_ans, trainings, t_ans)
pickle.dump(bind, valid_data)
valid_data.close()


print("Data Processing is done.")
