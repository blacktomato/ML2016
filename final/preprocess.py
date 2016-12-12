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
#c = np.identity(97)
#t = np.identity(300)

document = {}

<<<<<<< HEAD
with open(sys.argv[1] + '/promoted_content.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        Ad[int(i[0])] = [int(i[1]), int(i[2]), int(i[3])]

np.save(sys.argv[1]+"/Ad_detail", Ad)
'''
c = np.identity(97)
t = np.identity(300)

Document = np.zeros((3000000, 397))
=======
>>>>>>> master
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
<<<<<<< HEAD
    Document[category[n][0]][:97] += c[rc_dict[category[n][1]]] * P_category[n] 
=======
    element = (rc_dict[category[n][1]], P_category[n])
    if (category[n][0] in document):
        document[category[n][0]].append([element])
    else:
        document[category[n][0]] = [element]
>>>>>>> master

print("Preprocessing the topic")
n = 0
with open(sys.argv[1] + '/documents_topics.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
<<<<<<< HEAD
        Document[int(i[0])][97:397] += t[int(i[1])] * float(i[2]) 
=======
        element = (int(i[1])+97, float(i[2]))
        if (int(i[0]) in document):
            document[int(i[0])].append([element])
        else:
            document[int(i[0])] = [element]
>>>>>>> master
        n += 1

