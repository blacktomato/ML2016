#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : nn.py
 # Purpose : Implement nerual network to classify spam email or not
 # Creation Date : Sun 23 Oct 2016 02:53:55 PM CST
 # Last Modified : Wed 26 Oct 2016 10:57:13 AM CST
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


print("Data Processing is done.\nStart training...")

def E_function(w1, b1, w2, b2, testresult, testdata):
    offset = 0.001
    z_1 = np.dot(testdata, w1) + b1
    a_1 = 1 / (1+ math.e ** (-z_1))
    z_2 = np.sum( a_1 * w2, axis=1) + b2
    y = 1 / (1+ math.e ** (-z_2))
    cross_entropy = np.sum(testresult * np.log(y+offset, np.array([math.e]*4001)) + \
                    (1-testresult) * np.log((1-y+offset), np.array([math.e]*4001)))
    return -cross_entropy

#intial coefficient
weight_1 = np.zeros((57, 57))
bias_1 = np.zeros((1, 57))
weight_2 = np.zeros((1, 57))
bias_2 = 0
learning_time = 5000
#Regularization
Lambda = 0
#Adadelta
G_w1 = np.zeros((57, 57))
G_b1 = np.zeros((1, 57))
t_w1 = np.zeros((57, 57))
t_b1 = np.zeros((1, 57))
T_w1 = np.zeros((57, 57))
T_b1 = np.zeros((1, 57))
G_w2 = np.zeros((1, 57))
G_b2 = 0
t_w2 = np.zeros((1, 57))
t_b2 = 0
T_w2 = np.zeros((1, 57))
T_b2 = 0 
gamma = 0.9
epsilon = 10 ** -8

t = 1
while(True):
    z_1 = np.dot(data, weight_1) + bias_1
    a_1 = 1 / (1+ math.e ** (-z_1))
    z_2 = np.sum( a_1 * weight_2, axis=1) + bias_2
    y = 1 / (1+ math.e ** (-z_2))
    change = answer - y
    gradient_b2 = -1 * (change.sum())
    gradient_w2 = -1 * (np.sum(np.transpose(a_1) * change, axis=1))

    delta_1 = (a_1*(1-a_1)) * np.dot(np.transpose([change]), weight_2)
    gradient_b1 = -1 * np.sum(delta_1, axis=0)
    gradient_w1 = -1 * np.dot(np.transpose(data), delta_1)

    #gradient_1 adadelta
    G_w1 = gamma * G_w1 + (1 - gamma) * (gradient_w1 ** 2)
    G_b1 = gamma * G_b1 + (1 - gamma) * (gradient_b1 ** 2)
    t_w1 = -(((T_w1 + epsilon) ** 0.5) / ((G_w1 + epsilon) ** 0.5))  * gradient_w1
    t_b1 = -(((T_b1 + epsilon) ** 0.5) / ((G_b1 + epsilon) ** 0.5))  * gradient_b1
    T_w1 = gamma * T_w1 + (1 - gamma) * (t_w1 ** 2)
    T_b1 = gamma * T_b1 + (1 - gamma) * (t_b1 ** 2)
    weight_1 += t_w1
    bias_1 += t_b1
    
    #gradient_2 adadelta
    G_w2 = gamma * G_w2 + (1 - gamma) * (gradient_w2 ** 2)
    G_b2 = gamma * G_b2 + (1 - gamma) * (gradient_b2 ** 2)
    t_w2 = -(((T_w2 + epsilon) ** 0.5) / ((G_w2 + epsilon) ** 0.5))  * gradient_w2
    t_b2 = -(((T_b2 + epsilon) ** 0.5) / ((G_b2 + epsilon) ** 0.5))  * gradient_b2
    T_w2 = gamma * T_w2 + (1 - gamma) * (t_w2 ** 2)
    T_b2 = gamma * T_b2 + (1 - gamma) * (t_b2 ** 2)
    weight_2 += t_w2
    bias_2 += t_b2
    if (t % 1 == 0):
        print("The", t, "times__Cross Entropy:", E_function(weight_1, bias_1, weight_2, bias_2, answer, data) )
    if ( t > learning_time):
        print ("Logistic Regression training is done.")
        break
    t += 1

#output the model
model = open(sys.argv[2], "wb+")
bind = (weight_1, bias_1, weight_2, bias_2, mean, std_s)
pickle.dump(bind, model)
model.close()

print ("Training time:", t)
print("Cross Entropy:", E_function(weight_1, bias_1, weight_2, bias_2, answer, data) )
