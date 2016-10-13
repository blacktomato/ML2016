#!/usr/bin/env python
#coding=utf-8
##############################################################
 # File Name : validation.py
 # Purpose : Test learning efficiency for linear regression to predict the PM2.5
 # Creation Date : Sun 02 Oct 2016 14:17:35 CST
 # Last Modified : Fri 14 Oct 2016 03:16:33 AM CST
 # Created By : SL Chung
##############################################################
import numpy as np
import random

#Loss function L(w, b)
def loss_function(w, b, testresult, testdata, m, s, total):
    y_train = (np.sum(testdata * w, axis=1) + b) * s + m
    result = testresult - y_train
    return ((result ** 2).sum() / total ) ** 0.5

training_datas = [np.array(())]*10
training_results = [np.array(())]*10
ttraining_results = [np.array(())]*10
testing_datas = [np.array(())]*10
testing_results = [np.array(())]*10
ttesting_results = [np.array(())]*10

#For normalize
mean = np.array(())
std_d = np.array(())

            
print("Reading data file...")

mean = np.load("./data_validation/mean.npy")
std_d = np.load("./data_validation/std_sigma.npy")

for val in range(10):
    training_datas[val] = np.load("./data_validation/training_datas_" + str(val) + ".npy")
    training_results[val] = np.load("./data_validation/training_results_" + str(val) + ".npy")
    ttraining_results[val] = np.load("./data_validation/ttraining_results_" + str(val) + ".npy")
    testing_datas[val] = np.load("./data_validation/testing_datas_" + str(val) + ".npy")
    testing_results[val] = np.load("./data_validation/testing_results_" + str(val) + ".npy")
    ttesting_results[val] = np.load("./data_validation/ttesting_results_" + str(val) + ".npy")

print("Done")

print("Start training...")


#intial coefficient
weight = [np.zeros((1, 162))]*10
bias = [0]*10
learning_rate = 0.2
learning_time = 10000
#Regularization
Lambda = 10
G_w = np.zeros((1, 162))
#G_w2 = np.zeros((18, 9))
G_b = 0

last_l = 0
t = 1
while(True):
    l = 0;
    for val in range(10):
        change = ttraining_results[val] - bias[val] - np.sum((training_datas[val] * weight[val]), axis=1)
        b_w = change.sum()
        g_w = np.sum((np.transpose(training_datas[val]) * change), axis=1) - Lambda * weight[val]

        #gradient
        gradient_w = -2 * g_w
        gradient_b = -2 * b_w
        G_w += gradient_w ** 2
        G_b += gradient_b ** 2
        weight[val] = weight[val] - learning_rate * (1 / (G_w) ** 0.5 ) * gradient_w
        bias[val] = bias[val] - learning_rate * (1 / (G_b) ** 0.5 ) * gradient_b
        l += loss_function(weight[val], bias[val], \
            testing_results[val], testing_datas[val], mean[9], std_d[9], 564)
    print ("The " + str(t) + " times:  l =" ,l / 10)
    if (last_l != 0 and abs(last_l - l) < 0.0001 ):
        learning_rate *= 2
    last_l = l

    t += 1
    if ( t > learning_time):
        print ("Linear Regression training is done.")
        break

print("learning_rate:", learning_rate, "times:", learning_time, "lambda:", Lambda)

