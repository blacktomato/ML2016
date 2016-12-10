#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : preprocess.py
 # Purpose :
 # Creation Date : Fri 09 Dec 2016 00:59:20 CST
 # Last Modified : Sat 10 Dec 2016 02:41:17 PM CST
 # Created By : SL Chung
##############################################################
import sys
import numpy as np

'''
#              document_id  campaign_id  advertiser_id  
Ad = np.array([[         0,           0,             0]] * 573099)

with open(sys.argv[1] + '/promoted_content.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        Ad[int(i[0])] = [int(i[1]), int(i[2]), int(i[3])]

np.save(sys.argv[1]+"/Ad_detail", Ad)
'''
c = np.identity(97)
t = np.identity(300)

document = np.array([[0.0]*397] * 3000000)
print("Preprocessing the category")
n = 0
category   = np.array([[        0,           0]] * 5481475)
P_category = np.array([[                   0.0]] * 5481475)
with open(sys.argv[1] + '/documents_categories.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        category[n] = [int(i[0]), int(i[1])]
        P_category[n] = float(i[2])
        n += 1

allcate = np.bincount(category[:, 1]).nonzero()[0]
c_dict = dict(enumerate(allcate))
rc_dict = dict((v, k) for k, v in c_dict.iteritems())

for n in range(len(category)):
    document[category[n][0]][:97] += c[rc_dict[category[n][1]]] * P_category[n] 

print("Preprocessing the topic")
n = 0
with open(sys.argv[1] + '/documents_topics.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        document[int(i[0])][97:397] += t[int(i[1])] * float(i[2]) 
        n += 1

'''
with open(sys.argv[1] + '/events.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        geo = i[5].rstrip("\n").replace(">", "")
        geo = geo.replace("-", "")
        plat = i[4]
        if(not plat.isdigit()):
            plat = "0"
        if len(geo) == 0:
            geo = "0"
        Event[int(i[0])] = [int(i[1], 16), int(i[2]), int(i[3]), int(plat),int(geo.lower(), 36)]
'''

