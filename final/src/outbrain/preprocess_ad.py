#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : preprocess_ad.py
 # Purpose : Preprocess promoted_content.csv
 # Creation Date : Sat 10 Dec 2016 09:05:21 AM GMT
 # Last Modified : Sun 11 Dec 2016 01:32:05 PM CST
 # Created By : SL Chung
##############################################################
import sys
import numpy as np

#                document_id  campaign_id  advertiser_id  
# Ad = np.array([[         0,           0,             0]] * 573099)
Ad = np.zeros((573099, 3)).astype('int64')

with open(sys.argv[1] + '/promoted_content.csv') as fp:
    next(fp)
    for line in fp:
        i = line.split(",")
        Ad[int(i[0])] = [int(i[1]), int(i[2]), int(i[3])]

# np.save(sys.argv[2] + "/ad_nparray", Ad)
