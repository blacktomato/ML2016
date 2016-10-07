#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name :
 # Purpose :
 # Creation Date : Fri 07 Oct 2016 02:23:25 PM CST
 # Last Modified : Fri 07 Oct 2016 02:37:57 PM CST
 # Created By : SL Chung
##############################################################
import random as r

Ra = open("r.csv", "w+")

l = r.sample(range(471), 471)
for i in l:
    
