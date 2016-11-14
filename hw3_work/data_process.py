#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : cnn.py
 # Purpose : To classify the image
 # Creation Date : Fri 04 Nov 2016 10:41:51 AM CST
 # Last Modified : Sun 13 Nov 2016 00:35:44 CST
 # Created By : SL Chung
##############################################################
import numpy as np
import pickle
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation

#load the label and unlabel data
label_file = open(sys.argv[1] + 'all_label.p', "rb")
unlabel_file = open(sys.argv[1] + 'all_unlabel.p', "rb")
test_file = open(sys.argv[1] + 'test.p', "rb")
pre_label = pickle.load(label_file)
pre_unlabel = pickle.load(unlabel_file)
pre_test = pickle.load(test_file)
label_file.close()
unlabel_file.close()
test_file.close()

label = np.asarray(pre_label)           #10-500-3072
unlabel = np.asarray(pre_unlabel)       #45000-3072
test = np.asarray(pre_test['data'])     #10000

np.save("./Label", label)
np.save("./Unlabel", unlabel)
np.save("./Test", test)
