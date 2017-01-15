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

print("Store in document")
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


print("Preprocessing event")
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


print("Preprocessing ad")
Ad = np.zeros((573099, 3)).astype('int64')

with open(sys.argv[1] + '/promoted_content.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        Ad[int(i[0])] = [int(i[1]), int(i[2]), int(i[3])]

