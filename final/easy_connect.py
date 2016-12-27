#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : easy_connect.py
 # Purpose :
 # Creation Date : Tue 27 Dec 2016 11:18:54 PM CST
 # Last Modified : Tue 27 Dec 2016 11:32:03 PM CST
 # Created By : SL Chung
##############################################################
import pandas as pd

train = pd.read_csv("./clicks_train.csv")
d = pd.read_csv("./display.csv")
a = pd.read_csv("./ad.csv")

train = pd.merge(train, d, on="display_id")
train = train.drop("display_id", 1)
train["temp"] = train.clicked.astype(str).str.cat(train.d.adtype(str),sep=" ")
train = train.drop(["clicked","d"], 1)
train=pd.merge(train, a, on="ad_id")
train = train.drop("ad_id", 1)
train["i"] = train.temp.astype(str).str.cat(train.a.adtype(str))
train = train.drop(["temp","a"], 1)

train = train.sort_values(["display_id","ad_id"])
train = train.drop(["display_id","ad_id"], 1)

train.to_csv("train.csv", index=False)


test = pd.read_csv("./clicks_test.csv")

test = pd.merge(test, d, on="display_id")
del d
test = test.drop("display_id", 1)
test["temp"] = test.clicked.astype(str).str.cat(test.d.adtype(str),sep=" ")
test = test.drop(["clicked","d"], 1)
test=pd.merge(test, a, on="ad_id")
del a
test = test.drop("ad_id", 1)
test["i"] = test.temp.astype(str).str.cat(test.a.adtype(str))
test = test.drop(["temp","a"], 1)

test = test.sort_values(["display_id","ad_id"])
test = test.drop(["display_id","ad_id"], 1)

test.to_csv("test.csv", index=False)

