#!/usr/bin/env python
#coding=utf-8
##############################################################
 # File Name : predictorPM25.py
 # Purpose : Use linear regression to predict the PM2.5
 # Creation Date : Sun 02 Oct 2016 14:17:35 CST
 # Last Modified : Wed 05 Oct 2016 01:42:46 AM CST
 # Created By : SL Chung
##############################################################
import numpy as np

#Loss function L(w, b)
def loss_function(w, b, testresult, testdata, m, s):
    temp = np.zeros((1, len(testresult)));
    for i in range(len(testresult)):
        temp[0][i] = testresult[i] - (((w * testdata[i] ).sum() + b) * s + m)
    return ((temp ** 2).sum() / 240) ** 0.5

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

#use first 14 data in train_file
#and remain 1 for test
training_data = []     #create 3360 datas
training_result = np.array(())
testing_data = []
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

    if (d % 20 != 0):
        twdaycontainer = np.hstack((twdaycontainer, np.array(subday)))
    elif (d == 0):
        twdaycontainer = subday
    else:
        for i in range(20 * 24 - 10):
            if (i >= 480 - 20 - 9 ):
                testing_result = np.hstack((testing_result, np.array(twdaycontainer[9][i + 9])))
                data = twdaycontainer[:, i:i+9]
                testing_data.append(data)
            else:
                training_result = np.hstack((training_result, np.array(twdaycontainer[9][i + 9])))
                data = twdaycontainer[:, i:i+9]
                training_data.append(data)
        twdaycontainer = subday

for i in range(18):
    mean = np.hstack((mean, np.array(tempday[i].sum() / 5760 )))
    std_d = np.hstack((std_d, np.array(tempday_2[i].sum() / 5760 )))
std_d = (std_d - mean ** 2) ** 0.5

total = len(training_data)

training_result = (training_result - mean[9]) / std_d[9]
for i in range(total):
    training_data[i] = np.transpose((np.transpose(training_data[i]) - mean) / std_d)
for j in range(len(testing_data)):
    testing_data[j] = np.transpose((np.transpose(testing_data[j]) - mean) / std_d)


#intial coefficient
weight = (2 * np.random.random_sample((18, 9)) - 1) / 10000
bias = 0.001
learning_rate = 0.15
learning_time = 200
G_w = np.zeros((18, 9))
#G_w2 = np.zeros((18, 9))
G_b = 0

for t in range(learning_time):
    b_w = 0;
    g_w = np.zeros((18, 9))
    g_w2 = np.zeros((18, 9))
    for i in range(total):
        change = training_result[i] - ((weight * training_data[i]).sum() + bias)
        b_w += change
        g_w += (change * training_data[i])
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

    print ("The " + str(t) + " times:", loss_function(weight, bias, \
            testing_result, testing_data, mean[9], std_d[9]))

ftest_data = []
for day in final_test_days:
    data = np.zeros((18, 9))
    for row in range(18):
        for col in range(9):
            data[row][col] = day[row][col]
    ftest_data.append(data)
for i in range(len(ftest_data)):
    ftest_data[i] = np.transpose((np.transpose(ftest_data[i]) - mean) / std_d)



#Submission file
def DONE():
    Kaggle = open("For_Kaggle.csv", "w+")
    Kaggle.write("id,value\n")
    for i in range(len(ftest_data)):
        line = "id_" + str(i) + "," + str(((weight * ftest_data[i]).sum() + bias) * std_d[9] + mean[9]) + "\n"
        Kaggle.write(line)
    Kaggle.close()
DONE()
print ("The " + str(t) + " times:", loss_function(weight, bias, testing_result, testing_data, mean[9], std_d[9]))
