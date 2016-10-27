#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : nn.py
 # Purpose : Implement nerual network to classify spam email or not
 # Creation Date : Sun 23 Oct 2016 02:53:55 PM CST
 # Last Modified : Thu 27 Oct 2016 01:15:59 PM CST
 # Created By : SL Chung
##############################################################
import math
import numpy as np
import sys
import random
import pickle

valid_data = open(sys.argv[1], "rb")
(validations, v_ans, trainings, t_ans) = pickle.load(valid_data)
valid_data.close()

def E_function(w1, b1, w2, b2, testresult, testdata):
    offset = 0.0001
    z_1 = np.dot(testdata, w1) + b1
    a_1 = 1 / (1+ math.e ** (-z_1))
    z_2 = np.sum( a_1 * w2, axis=1) + b2
    y = 1 / (1+ math.e ** (-z_2))
    cross_entropy = np.sum(testresult * np.log(y+offset, np.array([math.e]*400)) + \
                    (1-testresult) * np.log((1-y+offset), np.array([math.e]*400)))
    return -cross_entropy

#intial coefficient
node = int(sys.argv[3])
weight_1 = [np.zeros((57, node))]*10
bias_1 = [np.zeros((1, node))]*10
weight_2 = [np.zeros((1, node))]*10
bias_2 = [0]*10
learning_time = int(sys.argv[2])
#Adadelta
G_w1 = [np.zeros((57, node))]*10
G_b1 = [np.zeros((1, node))]*10
t_w1 = [np.zeros((57, node))]*10
t_b1 = [np.zeros((1, node))]*10
T_w1 = [np.zeros((57, node))]*10
T_b1 = [np.zeros((1, node))]*10
G_w2 = [np.zeros((1, node))]*10
G_b2 = [0]*10
t_w2 = [np.zeros((1, node))]*10
t_b2 = [0]*10
T_w2 = [np.zeros((1, node))]*10
T_b2 = [0]*10
gamma = 0.9
epsilon = 10 ** -8

t = 0
while(True):
    t += 1
    CE = np.array([0.0] * 10)
    for i in range(10):
        z_1 = np.dot(trainings[i], weight_1[i]) + bias_1[i]
        a_1 = 1 / (1+ math.e ** (-z_1))
        z_2 = np.sum( a_1 * weight_2[i], axis=1) + bias_2[i]
        y = 1 / (1+ math.e ** (-z_2))
        change = a=t_ans[i] - y
        gradient_b2 = -1 * (change.sum())
        gradient_w2 = -1 * (np.sum(np.transpose(a_1) * change, axis=1))

        delta_1 = (a_1*(1-a_1)) * np.dot(np.transpose([change]), weight_2[i])
        gradient_b1 = -1 * np.sum(delta_1, axis=0)
        gradient_w1 = -1 * np.dot(np.transpose(trainings[i]), delta_1)

        #gradient_1 adadelta
        G_w1[i] = gamma * G_w1[i] + (1 - gamma) * (gradient_w1 ** 2)
        G_b1[i] = gamma * G_b1[i] + (1 - gamma) * (gradient_b1 ** 2)
        t_w1[i] = -(((T_w1[i] + epsilon) ** 0.5) / ((G_w1[i] + epsilon) ** 0.5))  * gradient_w1
        t_b1[i] = -(((T_b1[i] + epsilon) ** 0.5) / ((G_b1[i] + epsilon) ** 0.5))  * gradient_b1
        T_w1[i] = gamma * T_w1[i] + (1 - gamma) * (t_w1[i] ** 2)
        T_b1[i] = gamma * T_b1[i] + (1 - gamma) * (t_b1[i] ** 2)
        weight_1[i] += t_w1[i]
        bias_1[i] += t_b1[i]
        
        #gradient_2 adadelta
        G_w2[i] = gamma * G_w2[i] + (1 - gamma) * (gradient_w2 ** 2)
        G_b2[i] = gamma * G_b2[i] + (1 - gamma) * (gradient_b2 ** 2)
        t_w2[i] = -(((T_w2[i] + epsilon) ** 0.5) / ((G_w2[i] + epsilon) ** 0.5))  * gradient_w2
        t_b2[i] = -(((T_b2[i] + epsilon) ** 0.5) / ((G_b2[i] + epsilon) ** 0.5))  * gradient_b2
        T_w2[i] = gamma * T_w2[i] + (1 - gamma) * (t_w2[i] ** 2)
        T_b2[i] = gamma * T_b2[i] + (1 - gamma) * (t_b2[i] ** 2)
        weight_2[i] += t_w2[i]
        bias_2[i] += t_b2[i]

        if (t == learning_time):
            CE[i] = E_function(weight_1[i], bias_1[i], weight_2[i], bias_2[i], v_ans[i], validations[i])
    if (t == learning_time):
        mean_CE = np.sum(CE) / float(10)
        vari_CE = np.sum((CE - mean_CE)**2) / float(10)
        print("The", t, "times__mean_CE:", mean_CE, "vari_CE:", vari_CE )
        print ("Neural Network training is done.")
        break
