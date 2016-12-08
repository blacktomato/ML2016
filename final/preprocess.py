#!/usr/bin/env python3
# coding=utf-8
##############################################################
 # File Name : preprocess.py
 # Purpose :
 # Creation Date : Fri 09 Dec 2016 00:59:20 CST
 # Last Modified : Fri 09 Dec 2016 01:26:54 CST
 # Created By : SL Chung
##############################################################
import sys
import numpy as np


#              document_id  campaign_id  advertiser_id  
Ad = np.array([[         0,           0,             0]] * 573099)

with open('promoted_content.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        Ad[int(i[0])] = [int(i[1]), int(i[2]), int(i[3])]

np.save("Ad_detail", Ad)

