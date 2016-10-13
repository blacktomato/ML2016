#!/usr/bin/env python
#coding=utf-8
##############################################################
 # File Name : predictorPM25.py
 # Purpose : Use linear regression to predict the PM2.5
 # Creation Date : Sun 02 Oct 2016 14:17:35 CST
 # Last Modified : Thu 13 Oct 2016 09:33:10 PM CST
 # Created By : SL Chung
##############################################################
import numpy as np
import random

#Loss function L(w, b)
def loss_function(w, b, testresult, testdata, m, s, total):
    y_train = (np.sum(testdata * w, axis=1) + b) * s + m
    result = testresult - y_train
    return ((result ** 2).sum() / total ) ** 0.5

train_file = open('./data/train.csv', 'r', encoding='utf-8', errors='ignore')
test_file = open('./data/test_X.csv', 'r', encoding='utf-8', errors='ignore')
train_data = train_file.read().splitlines()
final_test = test_file.read().splitlines()

#Remove the header
train_data = train_data[1::]
train_days = []
onetenth_r = []
#Trim the data
day = np.zeros((18, 24)) 
for i in range(len(train_data)):
    #establish a day
    row = i % 18
    if row == 0 and i != 0:
        train_days.append(day)
        day = np.zeros((18, 24))    #start a new day
    #set string to number and NR to 0
    data_element = train_data[i].split(',')[3::]
    
    if row == 10:
        for j in range(24):
            if data_element[j] == 'NR':
                day[row][j] = float(0)
            else:
                day[row][j] = float(data_element[j])
    else:
        for j in range(24):
            day[row][j] = float(data_element[j])
    #For the last day
    if i == len(train_data) - 1:
        train_days.append(day)

#final test
final_test_days = []
day = np.zeros((18, 9)) 
for i in range(len(final_test)):
    #establish a day
    row = i % 18
    if row == 0 and i != 0:
        final_test_days.append(day)
        day = np.zeros((18, 9))    #start a new day
    #set string to number and NR to 0
    data_element = final_test[i].split(',')[2::]
    
    if row == 10:
        for j in range(9):
            if data_element[j] == 'NR':
                day[row][j] = float(0)
            else:
                day[row][j] = float(data_element[j])
    else:
        for j in range(9):
            day[row][j] = float(data_element[j])
    #For the last day
    if i == len(final_test) - 1:
        final_test_days.append(day)


training_datas = np.array(())
ttraining_results = np.array(())

#For normalize
mean = np.array(())
std_d = np.array(())
tempday = np.zeros((18, 24))
tempday_2 = np.zeros((18, 24))
twdaycontainer = np.zeros((18, 24))

first_training = True
for d in range(len(train_days)):
    day = train_days[d]
    tempday += day
    tempday_2 += day ** 2

for i in range(18):
    mean = np.hstack((mean, np.array(tempday[i].sum() / 5760 )))
    std_d = np.hstack((std_d, np.array(tempday_2[i].sum() / 5760 )))
std_d = (std_d - mean ** 2) ** 0.5

print("Processing Data...")

for d in range(len(train_days)):
    day = train_days[d]
    day = np.transpose( (np.transpose(day) - mean) / std_d )

    if (d % 20 != 0 and d % 20 != 19):
        twdaycontainer = np.hstack((twdaycontainer, np.array(day)))
    elif (d % 20 == 0):
        twdaycontainer = day
    else:
        twdaycontainer = np.hstack((twdaycontainer, np.array(day)))
        #choose twenty days per month
        for j in range(471):
            data = np.array(())
            ttraining_results = np.hstack((ttraining_results, np.array(twdaycontainer[9][j + 9])))
            data = np.transpose(twdaycontainer[:, j:j+9]).reshape(1, 162)
            if (d == 19 and first_training):
                first_training = False
                training_datas = np.hstack((training_datas, data[0]))
            else:
                training_datas = np.vstack((training_datas, data[0]))
            
training_results = ttraining_results * std_d[9] + mean[9]
            

print("Data Processing is done.\nStart training...")

#intial coefficient
weight = np.zeros((1, 162))
bias = 0
learning_rate = 0.05
learning_time = 2000
#Regularization
Lambda = 0
G_w = np.zeros((1, 162))
#G_w2 = np.zeros((18, 9))
G_b = 0

t = 1
while(True):
    change = ttraining_results - bias - np.sum((training_datas * weight), axis=1)
    b_w = change.sum()
    g_w = np.sum((np.transpose(training_datas) * change), axis=1) - Lambda * weight

    #gradient
    gradient_w = -2 * g_w
    gradient_b = -2 * b_w
    G_w += gradient_w ** 2
    G_b += gradient_b ** 2
    weight = weight - learning_rate * (1 / (G_w) ** 0.5 ) * gradient_w
    bias = bias - learning_rate * (1 / (G_b) ** 0.5 ) * gradient_b
    t += 1
    if ( t > learning_time):
        print ("Linear Regression training is done.")
        break
t = 0
ftest_data = np.array(())
for day in final_test_days:
    data = np.transpose(day).reshape(1, 162)
    t += 1
    if t == 1:
        ftest_data = np.hstack((ftest_data, data[0]))
    else:
        ftest_data = np.vstack((ftest_data, data[0]))


ftest_data = (ftest_data - np.tile(mean, 9)) / np.tile(std_d, 9)

#Submission file
def DONE():
    Kaggle = open("For_Kaggle.csv", "w+")
    Kaggle.write("id,value\n")
    result = (np.sum(ftest_data * weight, axis=1) + bias) * std_d[9] + mean[9]
    for i in range(240):
        line = "id_" + str(i) + "," + str(result[i]) + "\n"
        Kaggle.write(line)
    Kaggle.close()
DONE()
print("learning_rate:", learning_rate, "times:", learning_time)
