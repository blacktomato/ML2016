#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : test.py
 # Purpose :Use the model to test
 # Creation Date : Mon 24 Oct 2016 04:31:52 PM CST
 # Last Modified : Mon 24 Oct 2016 05:46:53 PM CST
 # Created By : SL Chung
##############################################################
import math
import numpy as np
import sys
import random
import pickle

model = open(sys.argv[1], "rb")
(weight, bias, mean, std_s) = pickle.load(model)
model.close()

test_file = open(sys.argv[2], "r", encoding='utf-8', errors='ignore')
test_data = test_file.read().splitlines()

test = np.array(())
for i in range(len(test_data)):
    #remove id for each data
    data_element = test_data[i].split(',')[1::]
    test_temp = np.array(())
    #data processing
    for j in range(57):
        test_temp = np.hstack(( test_temp, np.array( float(data_element[j]) ) ))
    if (i == 0):
        test = np.hstack(( test, test_temp ))
    else:
        test = np.vstack(( test, test_temp ))

#Normalization
test = (test - mean) / std_s

print("Test Data Processing is done.\nStart testing...")

z_final = np.sum(test * weight, axis=1) + bias
test_answer = np.around(1/(1 + math.e ** (-z_final)))


#output file
output= open(sys.argv[3], "w+")
output.write("id,label\n")

for i in range(len(test_data)):
    line = str(i+1) + "," + str(int(test_answer[i])) + "\n"
    output.write(line)
output.close()

print("Output file:", sys.argv[3])
