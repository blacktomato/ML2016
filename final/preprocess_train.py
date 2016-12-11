#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : preprocess_train.py
 # Purpose : Preprocess click_train.csv
 # Creation Date : Sun 11 Dec 2016 01:33:31 PM CST
 # Last Modified : Sun 11 Dec 2016 03:10:27 PM CST
 # Created By : SL Chung
##############################################################
import sys
import numpy as np
from sklearn.utils import shuffle

is_train = np.hstack((np.array([ True]*69713385), np.array([False]*17428346) ))
shuffle(is_train, random_state = 0)

train_data = np.array([[0.0]*799] * 69713385 )
train_ans  = np.array([[0]*2]      * 69713385 )
valid_data = np.array([[0.0]*799] * 17428346 )
valid_ans  = np.array([[0]*2]      * 17428346 )

#reading Event    [23120127 :   4]
#reading Document [ 3000000 : 397]
#reading Ad       [  573099 :   3]

n = 0
train_n = 0
valid_n = 0 
with open(sys.argv[1] + '/click_train.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        
        event = Event[int(i[0])]
        ad = Ad[int(i[1])]
        display = np.hstack((Document[event[0]], event[1:]))
        ad      = np.hstack((Document[   ad[0]],    ad[1:]))
        
        if(is_train[n]):
            train_data[train_n] = np.hstack((display, ad))
            train_ans[train_n][int(i[2])] = 1
            train_n += 1
        else:
            valid_data[valid_n] = np.hstack((display, ad))
            valid_ans[valid_n][int(i[2])] = 1
            valid_n += 1
        
        n += 1
