#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : preprocess_event.py
 # Purpose :
 # Creation Date : Sat 10 Dec 2016 03:56:32 PM CST
 # Last Modified : Sat 10 Dec 2016 05:14:50 PM CST
 # Created By : SL Chung
##############################################################
import sys
import numpy as np
import datetime

Event = np.array([[0]*4] * 23120127)
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

np.save("event_nparray.npy", Event)

