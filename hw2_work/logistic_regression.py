#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : logistic_regression.py
 # Purpose : Implement logistic regression to classify spam email or not
 # Creation Date : Sun 23 Oct 2016 02:53:55 PM CST
 # Last Modified : Tue 25 Oct 2016 02:13:39 AM CST
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

def E_function(w, b, testresult, testdata):
    offset = 0.0001
    z = (np.sum(testdata * w, axis=1) + b)
    f_wb = 1 / (1+ math.e ** (-z))
    cross_entropy = np.sum(testresult * np.log(f_wb+offset, np.array([math.e]*4001)) + \
                    (1-testresult) * np.log((1-f_wb+offset), np.array([math.e]*4001)))
    return -cross_entropy

#intial coefficient
weight = np.zeros((1, 57))
bias = 0
learning_time = 100000
#Regularization
Lambda = 0
#Adadelta
G_w = np.zeros((1, 57))
G_b = 0
t_w = np.zeros((1, 57))
t_b = 0
T_w = np.zeros((1, 57))
T_b = 0
gamma = 0.99
epsilon = 10 ** -8

t = 1
while(True):
    z = np.sum(data * weight, axis=1) + bias
    f_wb = 1 / (1+ math.e ** (-z))
    change = answer - f_wb 
    gradient_b = -1 * (change.sum())
    gradient_w = -1 * (np.sum(np.transpose(data) * change, axis=1) - Lambda * weight)

    #gradient adadelta
    G_w = gamma * G_w + (1 - gamma) * (gradient_w ** 2)
    G_b = gamma * G_b + (1 - gamma) * (gradient_b ** 2)
    t_w = -(((T_w + epsilon) ** 0.5) / ((G_w + epsilon) ** 0.5))  * gradient_w
    t_b = -(((T_b + epsilon) ** 0.5) / ((G_b + epsilon) ** 0.5))  * gradient_b
    T_w = gamma * T_w + (1 - gamma) * (t_w ** 2)
    T_b = gamma * T_b + (1 - gamma) * (t_b ** 2)
    weight += t_w
    bias += t_b
    if (t % 100 == 0):
        print("The", t, "times__Cross Entropy:", E_function(weight, bias, answer, data) )
    if ( t > learning_time):
        print ("Logistic Regression training is done.")
        break
    t += 1

#output the model
model = open(sys.argv[2], "wb+")
bind = (weight, bias, mean, std_s)
pickle.dump(bind, model)
model.close()

print ("Training time:", t)
print("Cross Entropy:", E_function(weight, bias, answer, data) )
