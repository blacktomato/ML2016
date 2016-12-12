#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : preprocess.py
 # Purpose : preprocess the data related to the documents 
 # Creation Date : Fri 09 Dec 2016 00:59:20 CST
 # Last Modified : Sun 11 Dec 2016 01:32:58 PM CST
 # Created By : SL Chung
##############################################################
import sys
import numpy as np
import datetime
from sklearn.utils import shuffle
from numpy import genfromtxt

c = np.identity(97)
t = np.identity(300)

Document = np.zeros((3000000, 397))
print("Preprocessing the category")
n = 0
category   = np.zeros((5481475, 2)).astype('int64')
P_category = np.zeros(5481475)
with open(sys.argv[1] + '/documents_categories.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        category[n] = [int(i[0]), int(i[1])]
        P_category[n] = float(i[2])
        n += 1

print("Mapping category")
allcate = np.bincount(category[:, 1]).nonzero()[0]
c_dict = dict(enumerate(allcate))
rc_dict = dict((v, k) for k, v in c_dict.items())

print("Store in Document")
for n in range(len(category)):
    Document[category[n][0]][:97] += c[rc_dict[category[n][1]]] * P_category[n] 

print("Preprocessing the topic")
n = 0
with open(sys.argv[1] + '/documents_topics.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        Document[int(i[0])][97:397] += t[int(i[1])] * float(i[2]) 
        n += 1


print("Preprocessing Ad")
Ad = np.zeros((573099, 3)).astype('int64')

with open(sys.argv[1] + '/promoted_content.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        Ad[int(i[0])] = [int(i[1]), int(i[2]), int(i[3])]


print("Preprocessing Event")
Event = np.zeros((23120127, 4)).astype('int64')
with open(sys.argv[1] + '/events.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
#        geo = i[5].rstrip("\n").replace(">", "")
#        geo = geo.replace("-", "")
        plat = i[4]
        if(not plat.isdigit()):
            plat = "2"
#        if len(geo) == 0:
#            geo = "0"
        trans_date = datetime.datetime
        date = trans_date.fromtimestamp( (1465876799998 + int(i[3])) / 1000)

        Event[int(i[0])] = [int(i[2]), date.day % 7, date.hour, int(plat)]


is_train = np.arange(33749186)
shuffle(is_train, random_state = 0)

print("Loading clicks")
#with open(sys.argv[1] + '/clicks_train.csv') as fp:

click_data = genfromtxt(sys.argv[2] + '/clicks_train_small.csv',
                         delimiter=',', dtype='int64', skip_header=1)

#train_data        
print("Processing data")
click_train = click_data[is_train[:1687459]]
event = Event[click_train[:, 0]]
ad = Ad[click_train[:, 1]]
display = np.hstack((Document[event[:, 0]], event[:, 1:]))
ad      = np.hstack((Document[   ad[:, 0]],    ad[:, 1:]))
data    = np.hstack((display, ad))
ans     = np.vstack((click_train[:, 2], 1 - click_train[:, 2])).T 

train_data = data[is_train[:1349967]]
train_ans  =  ans[is_train[:1349967]]
            
valid_data = data[is_train[1349967:1687459]]
valid_ans  =  ans[is_train[1349967:1687459]]
