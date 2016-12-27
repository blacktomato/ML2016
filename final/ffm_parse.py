#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : ffm_parse.py
 # Purpose : Parse data for FFM
 # Creation Date : Sun 25 Dec 2016 05:11:35 PM CST
 # Last Modified : Tue 27 Dec 2016 08:57:07 PM CST
 # Created By : SL Chung
##############################################################
import sys
import numpy as np
import pandas as pd
import datetime

#parse the document category and topic
cat = pd.read_csv("./documents_categories.csv")
top = pd.read_csv("./documents_topics.csv")

#mapping category_id to 0~96
category_id = np.bincount(cat.category_id.as_matrix()).nonzero()[0]
category_id.sort()
newcat = np.arange(category_id.shape[0])
cat_mapping = np.transpose(np.vstack((category_id, newcat)))
cat_mapping = pd.DataFrame(cat_mapping, columns=["category_id", "new category_id"])
cat = pd.merge(cat, cat_mapping, on="category_id")
cat = cat.drop("category_id", 1) #0 means row, 1 means column
cat = cat[['document_id', 'new category_id', 'confidence_level']]
cat.columns = ['document_id', 'category_id', 'confidence_level']

cat = cat.sort_values(['document_id', 'category_id'])
top = top.sort_values(['document_id', 'topic_id'])

#document_id, category(96:0.34) / topic(299:0.345)
cat["category"]=cat.category_id.astype(str).str.cat(cat.confidence_level.astype(str), sep=":")
top["topic"]=top.topic_id.astype(str).str.cat(top.confidence_level.astype(str), sep=":")
cat = cat.drop(["category_id", "confidence_level"], 1)
top = top.drop(["topic_id", "confidence_level"], 1)

cat.to_csv("cate.csv", index=False)
top.to_csv("top.csv", index=False)

catfile = open("./cate_ffm.csv", "w+")
topfile = open("./top_ffm.csv", "w+")
catfile.write( 'document_id,d_category,a_category\n')
topfile.write( 'document_id,d_topic,a_topic\n')
with open('./cate.csv') as fc, open('./top.csv') as ft:
    next(fc)
    next(ft)
    catfile.write('1,')
    topfile.write('1,')

    #catfile
    document_id = 1;
    for_display = ""
    for_ad = ""
    for line in fc:
        i = line.split(",")
        if (document_id == int(i[0])):
            for_display += ' 0:' + str.rstrip(i[1], '\n') 
            for_ad += ' 5:' + str.rstrip(i[1], '\n')
        else:
            catfile.write(for_display + ','+ for_ad)
            for_display = ' 0:' + str.rstrip(i[1], '\n') 
            for_ad = ' 5:' + str.rstrip(i[1], '\n')
            for j in range(int(i[0]) - document_id):
                document_id += 1
                catfile.write('\n' + str(document_id) + ',')
                print(document_id, for_display)
    catfile.write(for_display + ','+ for_ad)

    #topfile
    document_id = 1;
    for_display = ""
    for_ad = ""
    for line in ft:
        i = line.split(",")
        if (document_id == int(i[0])):
            for_display += ' 1:' + str.rstrip(i[1], '\n') 
            for_ad += ' 6:' + str.rstrip(i[1], '\n')
        else:
            topfile.write(for_display + ','+ for_ad)
            for_display = ' 1:' + str.rstrip(i[1], '\n') 
            for_ad = ' 6:' + str.rstrip(i[1], '\n')
            for j in range(int(i[0]) - document_id):
                document_id += 1
                topfile.write('\n' + str(document_id) + ',')
    topfile.write(for_display + ','+ for_ad)


del cat
doc = pd.read_csv("./cate_ffm.csv")
top = pd.read_csv("./top_ffm.csv")
doc = pd.merge(doc, top, on="document_id", how='outer')
del top


doc = doc.replace(np.nan,'')     
doc["diplay"]=doc.d_category.astype(str).str.cat(doc.d_topic.astype(str), sep="")
doc["ad"]=doc.a_category.astype(str).str.cat(doc.a_topic.astype(str), sep="")
doc = doc.drop(["d_category", "d_topic", "a_category", "a_topic"], 1)
doc.to_csv("doc.csv", index=False)

#doc = pd.read_csv("doc.csv")
doc = doc.replace(np.nan,'')     

#Turn event.csv into:
#display_id,document_id,WeekHourPlatform
eventfile = open("./event_ffm.csv", "w+")
eventfile.write( 'display_id,document_id,whp\n')
with open('./events.csv') as fp:
    next(fp)
    #outfile.write('1,')
    for line in fp:
        i = line.split(",")
        plat = i[4]
        if(not plat.isdigit()):
            plat = "2"
        trans_date = datetime.datetime
        date = trans_date.fromtimestamp( (1465876799998 + int(i[3])) / 1000)
        inform = i[0] + ',' + i[2]+', 2:'+str(date.day % 7) + ':1 3:' + str(date.hour)+":1 4:"+ plat+":1\n"
        #print(inform)
        eventfile.write(inform)

#promoted_content.csv
#ad_id,ad_document_id,campaign_id,advertiser_id
adfile = open("./ad_ffm.csv", "w+")
adfile.write( 'ad_id,document_id,ca\n')
with open('./promoted_content.csv') as fp:
    next(fp)
    ad_id = 1;
    #outfile.write('1,')
    for line in fp:
        i = line.split(",")
        inform = i[0] + ',' + i[1]+', 7:'+ i[2]+ ':1 8:' + str.rstrip(i[3], '\n') + ':1\n'
        adfile.write(inform)

event = pd.read_csv("./event_ffm.csv")
ad = pd.read_csv("./ad_ffm.csv")

event = pd.merge(event, doc, on="document_id")
event = event.drop("ad", 1 )
event["d"] = event.display.astype(str).str.cat(event.whp.astype(str), sep="")
event = event.drop(["whp", "display"], 1 )
event = event.sort_values("display_id")
event.to_csv("display.csv", index=False)

ad = pd.merge(ad, doc, on="document_id")
ad = ad.drop("display", 1 )
del doc
ad["a"] = ad.ad.astype(str).str.cat(ad.ca.astype(str), sep="")
ad = ad.drop(["ca", "ad"], 1 )
ad = ad.sort_values("ad_id")
ad.to_csv("ad.csv", index=False)

#click_train.csv
#display_id,ad_id,clicked
