#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name :
 # Purpose :
 # Creation Date : Fri 07 Oct 2016 02:23:25 PM CST
 # Last Modified : Mon 10 Oct 2016 04:21:03 PM CST
 # Created By : SL Chung
##############################################################
import random as r

Ra = open("r.csv", "w+")

def shift(lis, n):
    return lis[n:] + lis[:n]

l = r.sample(range(471), 471)

for j in range(1,11):
    for i in l:
        Ra.write(",");
        Ra.write(str(i))
    Ra.write("\n")
    l = shift(l, 47)


Ra.close();
    
