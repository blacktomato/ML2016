#!/usr/bin/env python
#coding=utf-8
##############################################################
 # File Name : predictorPM25.py
 # Purpose : Use linear regression to predict the PM2.5
 # Creation Date : Sun 02 Oct 2016 14:17:35 CST
 # Last Modified : Thu 06 Oct 2016 02:33:49 CST
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
                data_element[j] = float(0)
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
                data_element[j] = float(0)
            else:
                data_element[j] = float(data_element[j])
    else:
        for j in range(9):
            data_element[j] = float(data_element[j])
    day.append(data_element)
    #For the last day
    if i == len(final_test) - 1:
        final_test_days.append(day)

training_data = np.array(())   
training_result = np.array(())
testing_data = np.array(())
testing_result = np.array(())

#For normalize
mean = np.array(())
std_d = np.array(())
tempday = np.zeros((18, 24))
tempday_2 = np.zeros((18, 24))
twdaycontainer = np.zeros((18, 24))

for d in range(len(train_days)):
    day = train_days[d]
    subday = np.zeros((18, 24))
    for i in range(18):
        for j in range(24):
            subday[i][j] = day[i][j]

    tempday += subday
    tempday_2 += subday ** 2
    if (d % 20 != 0 and d % 20 != 19):
        twdaycontainer = np.hstack((twdaycontainer, np.array(subday)))
    elif (d % 20 == 0):
        twdaycontainer = subday
    else:
        twdaycontainer = np.hstack((twdaycontainer, np.array(subday)))
        for i in range(20 * 24 - 9):
        #0 1 2   451(452)first_testing_data  470(471)last_testing_data     479(480)
            data = np.array(())
            if (i >= 480 - 20 - 9 ):
                testing_result = np.hstack((testing_result, np.array(twdaycontainer[9][i + 9])))
                #training_result = np.hstack((training_result, np.array(twdaycontainer[9][i + 9])))
                for j in range(1,10):
                    data = np.hstack((data, np.transpose(twdaycontainer[:, i+j])))
                if (d == 19 and i == 451):
                    testing_data = np.hstack((testing_data, data))
                    #training_data = np.vstack((training_data, data))
                else:
                    testing_data = np.vstack((testing_data, data))
                    #training_data = np.vstack((training_data, data))
            else:
                training_result = np.hstack((training_result, np.array(twdaycontainer[9][i + 9])))
                for j in range(1,10):
                    data = np.hstack((data, np.transpose(twdaycontainer[:, i+j])))
                if (d == 19 and i == 0):
                    training_data = np.hstack((training_data, data))
                else:
                    training_data = np.vstack((training_data, data))

for i in range(18):
    mean = np.hstack((mean, np.array(tempday[i].sum() / 5760 )))
    std_d = np.hstack((std_d, np.array(tempday_2[i].sum() / 5760 )))
std_d = (std_d - mean ** 2) ** 0.5


ttraining_result = (training_result - mean[9]) / std_d[9]
training_data = (training_data - np.tile(mean, 9)) / np.tile(std_d, 9)
testing_data = (testing_data - np.tile(mean, 9)) / np.tile(std_d, 9)
print(training_data.shape)
print(testing_data.shape)

#intial coefficient
weight = (2 * np.random.random_sample((1, 162)) - 1) / 10000
bias = (2 * random.random() - 1) / 10000
learning_rate = 0.15
learning_time = 2000
Lambda = 1
G_w = np.zeros((1, 162))
#G_w2 = np.zeros((18, 9))
G_b = 0
t = 1
while(True):
    change = ttraining_result - bias - np.sum((training_data * weight), axis=1)
    b_w = change.sum()
    g_w = np.sum((np.transpose(training_data) * change), axis=1)
    #g_w2 += (change * training_data[i] ** 2)

    #gradient
    gradient_w = -2 * g_w
    #gradient_w2 = -2 * g_w2
    gradient_b = -2 * b_w
    G_w += gradient_w ** 2
    #G_w2 += gradient_w2 ** 2
    G_b += gradient_b ** 2
    weight = weight - learning_rate * (1 / (G_w + 0.00000001) ** 0.5 ) * gradient_w
    #weight_2 = weight_2 - learning_rate * (1 / (G_w2 + 0.00000001) ** 0.5 ) * gradient_w2
    bias = bias - learning_rate * (1 / (G_b + 0.00000001) ** 0.5 ) * gradient_b
    if (t % 10 == 0):
        l = loss_function(weight, bias, testing_result, testing_data, mean[9], std_d[9], 240)
        print ("The " + str(t) + " times: l =",l)
    t += 1
    if ( t > learning_time):
        print ("fuck")
        break
t = 0
ftest_data = np.array(())
for day in final_test_days:
    data = np.array(())
    for row in range(18):
        for col in range(9):
            data = np.hstack((data, np.array(day[row][col])))
    t += 1
    if t == 1:
        ftest_data = np.hstack((ftest_data, data))
    else:
        ftest_data = np.vstack((ftest_data, data))

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
