#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : predictorPM25.py
 # Purpose : Use linear regression to predict the PM2.5
 # Creation Date : Sun 02 Oct 2016 14:17:35 CST
 # Last Modified : Mon 03 Oct 2016 02:04:34 CST
 # Created By : SL Chung
##############################################################
import numpy as np

train_file = open('./data/train.csv', 'r')
train_data = train_file.read().splitlines()

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


