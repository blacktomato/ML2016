#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : logistic_regression.py
 # Purpose : Implement logistic regression to classify spam email or not
 # Creation Date : Sun 23 Oct 2016 02:53:55 PM CST
 # Last Modified : Mon 24 Oct 2016 02:28:33 AM CST
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
    
print("Data Processing is done.\nStart training...")

def likelihood_function(w, b, testresult, testdata):
    z = (np.sum(testdata * w, axis=1) + b)
    f_wb = 1 / (1+ math.e ** (-z))
    p = np.prod(testresult * f_wb + (1-testresult) * (1-f_wb))
    cross_entropy = np.sum(testresult * np.log(f_wb, np.array([math.e]*4001)) + \
                    (1-testresult) * np.log((1-f_wb), np.array([math.e]*4001)))
    print("Cross Entropy:", -cross_entropy)
    return p

#intial coefficient
weight = np.zeros((1, 57))
bias = 0
learning_rate = 0.0001
learning_time = 10000
#Regularization
Lambda = 0
G_w = np.zeros((1, 57))
G_b = 0

t = 1
while(True):
    z = np.sum(data * weight, axis=1) + bias
    f_wb = 1 / (1+ math.e ** (-z))
    change = answer - f_wb 
    b_w = change.sum()
    g_w = np.sum(np.transpose(data) * change, axis=1) - Lambda * weight

    #gradient
    gradient_w = -g_w
    gradient_b = -b_w
    G_w += gradient_w ** 2
    G_b += gradient_b ** 2
    weight = weight - learning_rate * (1 / (G_w) ** 0.5 ) * gradient_w
    bias = bias - learning_rate * (1 / (G_b) ** 0.5 ) * gradient_b
    print("The", t, "times__likelihood: ", likelihood_function(weight, bias, answer, data) )
    if ( t > learning_time):
        print ("Logistic Regression training is done.")
        break
    t += 1
