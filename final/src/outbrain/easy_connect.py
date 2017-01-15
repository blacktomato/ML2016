#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : easy_connect.py
 # Purpose :
 # Creation Date : Tue 27 Dec 2016 11:18:54 PM CST
 # Last Modified : Thu 29 Dec 2016 03:47:50 PM CST
 # Created By : SL Chung
##############################################################
import pandas as pd

train = pd.read_csv("./clicks_train.csv")
d = pd.read_csv("./display.csv")
a = pd.read_csv("./ad.csv")

train = pd.merge(train, d, on="display_id")
train = train.drop("display_id", 1)
train["temp"] = train.clicked.astype(str).str.cat(train.d.astype(str),sep=" ")
train = train.drop(["clicked","d"], 1)
train=pd.merge(train, a, on="ad_id")
train = train.drop("ad_id", 1)
train["i"] = train.temp.astype(str).str.cat(train.a.astype(str))
train = train.drop(["temp","a"], 1)

train = train.sort_values(["display_id","ad_id"])
train = train.drop(["display_id","ad_id"], 1)

train.to_csv("train.csv", index=False)


test = pd.read_csv("./clicks_test.csv")

test = pd.merge(test, d, on="display_id")
del d
test=pd.merge(test, a, on="ad_id")
del a
#add "0 " in the front of i
test["i"] = ("0 " + test.d.astype(str)).str.cat(test.a.astype(str))
test = test.drop(["d","a"], 1)

test = test.sort_values(["display_id","ad_id"])
test = test.drop(["display_id","ad_id"], 1)

test.to_csv("test.csv", index=False)

