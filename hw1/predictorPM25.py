#!/usr/bin/env python
#coding=utf-8
##############################################################
 # File Name : predictorPM25.py
 # Purpose : Use linear regression to predict the PM2.5
 # Creation Date : Sun 02 Oct 2016 14:17:35 CST
 # Last Modified : Tue 04 Oct 2016 09:34:10 AM CST
 # Created By : SL Chung
##############################################################
import numpy as np
from scipy.misc import derivative

#Loss function L(w, b)
def loss_function(w, b, testresult, testdata):
    temp = np.zeros((1, len(testresult)));
    for i in range(len(testresult)):
        temp[0][i] = testresult[i] - ((w * testdata[i]).sum() + b)
    return (temp ** 2).sum()

train_file = open('./data/train.csv', 'r', encoding='utf-8', errors='ignore')
test_file = open('./data/test_X.csv', 'r', encoding='utf-8', errors='ignore')
train_data = train_file.read().splitlines()
final_test = test_file.read().splitlines()

#Remove the header
train_data = train_data[1::]
train_days = []
#Trim the data
day = []
for i in range(len(train_data)):
    #establish a day
    if i % 18 == 0 and i != 0:
        train_days.append(day)
        day = []    #start a new day
    #set string to number and NR to 0
    data_element = train_data[i].split(',')[3::]
    
    if i % 18 == 10:
        for j in range(24):
            if data_element[j] == 'NR':
                data_element[j] = 0
            else:
                data_element[j] = float(data_element[j])
    else:
        for j in range(24):
            data_element[j] = float(data_element[j])
    day.append(data_element)
    #For the last day
    if i == len(train_data) - 1:
        train_days.append(day)

#final test
final_test_days = []
day = []
for i in range(len(final_test)):
    #establish a day
    if i % 18 == 0 and i != 0:
        final_test_days.append(day)
        day = []    #start a new day
    #set string to number and NR to 0
    data_element = final_test[i].split(',')[2::]
    
    if i % 18 == 10:
        for j in range(9):
            if data_element[j] == 'NR':
                data_element[j] = 0
            else:
                data_element[j] = float(data_element[j])
    else:
        for j in range(9):
            data_element[j] = float(data_element[j])
    day.append(data_element)
    #For the last day
    if i == len(final_test) - 1:
        final_test_days.append(day)

#intial coefficient
weight = np.ones((18, 9)) / 10000
bias = 1
#use first 14 data in train_file
#and remain 1 for test
training_data = []     #create 3360 datas
training_result = np.array(())
testing_data = []
testing_result = np.array(())

for day in train_days:
    for i in range(15):
        if (i == 14):
            testing_result = np.hstack((testing_result, np.array(day[9][23])))
            data = np.zeros((18, 9))
            for row in range(18):
                for col in range(9):
                    data[row][col] = day[row][col]
            testing_data.append(data)
        else:
            training_result = np.hstack((training_result, np.array(day[9][9 + i])))
            data = np.zeros((18, 9))
            for row in range(18):
                for col in range(9):
                    data[row][col] = day[row][col]
            training_data.append(data)

total = len(training_data)
learning_rate = 100
learning_time = 2000
G_w = np.zeros((18, 9))
G_b = 0
for t in range(learning_time):
    b_w = 0;
    g_w = np.zeros((18, 9))
    for i in range(total):
        b_w = b_w + training_result[i] - ((weight * training_data[i]).sum() + bias)
        g_w = g_w + (training_result[i] - ((weight * training_data[i]).sum() + bias)) * training_data[i]

    #gradient
    gradient_w = -2 * g_w
    gradient_b = 2 * b_w
    G_w += gradient_w ** 2
    G_b += gradient_b ** 2
    weight = weight - learning_rate * (1 / (G_w + 0.00000001) ** 0.5 ) * gradient_w
    bias = bias - learning_rate * (1 / (G_b + 0.00000001) ** 0.5 ) * gradient_b

    print ("The " + str(t) + " times:", loss_function(weight, bias, testing_result, testing_data))

ftest_data = []
for day in final_test_days:
    data = np.zeros((18, 9))
    for row in range(18):
        for col in range(9):
            data[row][col] = day[row][col]
    ftest_data.append(data)

#Submission file
def DONE():
    Kaggle = open("For_Kaggle.csv", "w+")
    Kaggle.write("id,value\n")
    for i in range(len(ftest_data)):
        line = "id_" + str(i) + "," + str((weight * ftest_data[i]).sum() + bias) + "\n"
        Kaggle.write(line)
    Kaggle.close()
DONE()

