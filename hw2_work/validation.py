#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : validation.py
 # Purpose :
 # Creation Date : Mon 24 Oct 2016 08:35:29 PM CST
 # Last Modified : Mon 24 Oct 2016 11:26:33 PM CST
 # Created By : SL Chung
##############################################################
import math
import numpy as np
import sys
import random

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

print("Data Processing is done.\nStart training...")

def E_function(w, b, testresult, testdata):
    offset = 0.0001
    z = (np.sum(testdata * w, axis=1) + b)
    f_wb = 1 / (1+ math.e ** (-z))
    cross_entropy = np.sum(testresult * np.log(f_wb+offset, np.array([math.e]*400)) + \
                    (1-testresult) * np.log((1-f_wb+offset), np.array([math.e]*400)))
    return -cross_entropy

#intial coefficient
weight = [np.zeros((1, 57))]*10
bias = [0]*10
learning_time = 100000
#Regularization
Lambda = 0
#Adadelta
G_w = [np.zeros((1, 57))]*10
G_b = [0]*10
t_w = [np.zeros((1, 57))]*10
t_b = [0]*10
T_w = [np.zeros((1, 57))]*10
T_b = [0]*10
gamma = 0.99
epsilon = 10 ** -8 

t = 1
while(True):
    CE = np.array([0.0] * 10)
    for i in range(10):
        z = np.sum(trainings[i] * weight[i], axis=1) + bias[i]
        f_wb = 1 / (1+ math.e ** (-z))
        change = t_ans[i] - f_wb 
        gradient_b = -1 * (change.sum())
        gradient_w = -1 * (np.sum(np.transpose(trainings[i]) * change, axis=1))

        #gradient adadelta
        G_w[i] = gamma * G_w[i] + (1 - gamma) * (gradient_w ** 2)
        G_b[i] = gamma * G_b[i] + (1 - gamma) * (gradient_b ** 2)
        t_w[i] = -(((T_w[i] + epsilon) ** 0.5) / ((G_w[i] + epsilon) ** 0.5))  * gradient_w
        t_b[i] = -(((T_b[i] + epsilon) ** 0.5) / ((G_b[i] + epsilon) ** 0.5))  * gradient_b
        T_w[i] = gamma * T_w[i] + (1 - gamma) * (t_w[i] ** 2)
        T_b[i] = gamma * T_b[i] + (1 - gamma) * (t_b[i] ** 2)
        weight[i] += t_w[i]
        bias[i] += t_b[i]

        if (t % 100 == 0):
            CE[i] = E_function(weight[i], bias[i], v_ans[i], validations[i])
    mean_CE = np.sum(CE) / float(10)
    vari_CE = np.sum((CE - mean_CE)**2) / float(10)
    if (t % 100 == 0):
        print("The", t, "times__mean_CE:", mean_CE, "vari_CE:", vari_CE )
        #print("The", t, "times__mean_CE:", CE)
    if ( t > learning_time):
        print ("Logistic Regression training is done.")
        break
    t += 1

