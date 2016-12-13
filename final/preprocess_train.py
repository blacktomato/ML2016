#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : preprocess_train.py
 # Purpose : Preprocess click_train.csv
 # Creation Date : Sun 11 Dec 2016 01:33:31 PM CST
 # Last Modified : Tue 13 Dec 2016 12:50:58 AM CST
 # Created By : SL Chung
##############################################################
import sys
import numpy as np
from sklearn.utils import shuffle
from numpy import genfromtxt

# is_train = np.hstack((np.ones(69713385), np.zeros(17428346))).astype('bool')
#is_train = np.hstack((np.ones(26999349), np.zeros(6749837))).astype('bool')
is_train = np.arange(33749186)
shuffle(is_train, random_state = 0)

# train_data = np.zeros((69713385, 799))
# train_ans  = np.zeros((69713385, 2))
# valid_data = np.zeros((17428346, 799))
# valid_ans  = np.zeros((17428346, 2))
train_data = np.zeros((26999349, 799))
train_ans  = np.zeros((26999349, 2)).astype('int64')
valid_data = np.zeros((6749837, 799))
valid_ans  = np.zeros((6749837, 2)).astype('int64')

#reading Event    [23120127 :   4]
#reading Document [ 3000000 : 397]
#reading Ad       [  573099 :   3]

print("Processing data")
#with open(sys.argv[1] + '/clicks_train.csv') as fp:

click_data = genfromtxt(sys.argv[2] + '/clicks_train_small.csv',
                         delimiter=',', dtype='int64', skip_header=1)

#train_data        
click_train = click_data[is_train[:3374918]]
event = Event[click_train[:, 0]]
ad = Ad[click_train[:, 1]]
display = np.hstack((Document[event[:, 0]], event[:, 1:]))
ad      = np.hstack((Document[   ad[:, 0]],    ad[:, 1:]))
data    = np.hstack((display, ad))
ans     = np.hstack((click_train[:, 2], 1 - click_train[:, 2])) 

data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

train_data = data[is_train[:2699934]]
train_ans  =  ans[is_train[:2699934]]
            
valid_data = data[is_train[2699934:3374918]]
valid_ans  =  ans[is_train[2699934:3374918]]
