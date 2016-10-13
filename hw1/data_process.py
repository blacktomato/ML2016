#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : data_process.py
 # Purpose : Create data used for validation
 # Creation Date : Fri 14 Oct 2016 12:42:54 AM CST
 # Last Modified : Fri 14 Oct 2016 12:54:33 AM CST
 # Created By : SL Chung
##############################################################
import numpy as np
import random

train_file = open('./data/train.csv', 'r', encoding='utf-8', errors='ignore')
r_file = open('./data/r.csv', 'r', encoding='utf-8', errors='ignore')
r_data = r_file.read().splitlines()
train_data = train_file.read().splitlines()

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


#For K-Validation 1 / 10
for i in range(len(r_data)):
    data_element = r_data[i].split(',')
    for j in range(len(data_element)):
        data_element[j] = int(data_element[j])
    onetenth_r.append(data_element)

training_datas = [np.array(())]*10
training_results = [np.array(())]*10
ttraining_results = [np.array(())]*10
testing_datas = [np.array(())]*10
testing_results = [np.array(())]*10
ttesting_results = [np.array(())]*10

#For normalize
mean = np.array(())
std_d = np.array(())
tempday = np.zeros((18, 24))
tempday_2 = np.zeros((18, 24))
twdaycontainer = np.zeros((18, 24))

first_training = [True]*10
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
        for val in range(10):
            all_data = onetenth_r[val]
            #For validation
            chosen_data = all_data[0:47]
            #For trainiing
            rest_data = all_data[47:]
            for i in chosen_data:
                data = np.array(())
                ttesting_results[val] = np.hstack((ttesting_results[val], np.array(twdaycontainer[9][i + 9])))
                data = np.transpose(twdaycontainer[:, i:i+9]).reshape(1, 162)
                if (d == 19 and i == chosen_data[0]):
                    testing_datas[val] = np.hstack((testing_datas[val], data[0]))
                else:
                    testing_datas[val] = np.vstack((testing_datas[val], data[0]))
            for j in rest_data:
                data = np.array(())
                ttraining_results[val] = np.hstack((ttraining_results[val], np.array(twdaycontainer[9][j + 9])))
                data = np.transpose(twdaycontainer[:, j:j+9]).reshape(1, 162)
                if (d == 19 and first_training[val]):
                    first_training[val] = False
                    training_datas[val] = np.hstack((training_datas[val], data[0]))
                else:
                    training_datas[val] = np.vstack((training_datas[val], data[0]))
            
for val in range(10):
    training_results[val] = ttraining_results[val] * std_d[9] + mean[9]
    testing_results[val] = ttesting_results[val] * std_d[9] + mean[9]
    np.save("./data_validation/training_datas_" + str(val), training_datas[val])
    np.save("./data_validation/training_results_" + str(val), training_results[val])
    np.save("./data_validation/ttraining_results_" + str(val), ttraining_results[val])
    np.save("./data_validation/testing_datas_" + str(val), testing_datas[val])
    np.save("./data_validation/testing_results_" + str(val), testing_results[val])
    np.save("./data_validation/ttesting_results_" + str(val), ttesting_results[val])

np.save("./data_validation/mean", mean)
np.save("./data_validation/std_sigma", std_d)

print("Data Processing is done and output.")

