#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : cluster.py
 # Purpose : clustering the title on the Stack OverFlow
 # Creation Date : Thu 24 Nov 2016 01:48:34 PM CST
 # Last Modified : Fri 02 Dec 2016 05:02:03 PM CST
 # Created By : SL Chung
##############################################################

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

start_time = time.time()

#Set the tf-idf vectorizer
vectorizer = CountVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2),
                             stop_words='english')
title_file = open( sys.argv[1] + "/title_StackOverflow.txt")

corpus = title_file.readlines()
title_file.close();

X = vectorizer.fit_transform(corpus)
#analyze = vectorizer.build_analyzer()
print("Features are total:", X.shape )

svd = TruncatedSVD(20)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

#svd2 = TruncatedSVD(20)
#lsa2 = make_pipeline(svd2, normalizer)
#X_lsa = lsa2.fit_transform(X) 
#tsne = TSNE(n_components=3, random_state=0)
#np.set_printoptions(suppress=True)
#pic = tsne.fit_transform(X_lsa) 

#svd2 = TruncatedSVD(3)
#visual = make_pipeline(svd2, normalizer)

#Array of the vectorized titles.
features = lsa.fit_transform(X) 
#pic = visual.fit_transform(X) 

print("Features are total:", features.shape )

n = 100
kmeans = KMeans(n_clusters=n , random_state=0).fit(features)

#kmeans2 = KMeans(n_clusters=20 ,n_init=1, random_state=0).fit(kmeans.cluster_centers_)
#for i in range(20):
#    merge_index = ((kmeans2.labels_ == i) * 1).nonzero()
    

print("Number in each cluster:", np.sort(np.bincount(kmeans.labels_) ))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#color = kmeans.labels_
#pic = np.transpose(pic)
#for i in range(n):
#    #index of same cluster
#    index = ((color==0)*1).nonzero()
#    #plt.scatter(pic[0][index], pic[1][index], color=str(float(i+30) / (n+50)))
#    ax.scatter(pic[0][index], pic[1][index], pic[2][index],c=str(float(i) /(n)))
#    color = color - 1
#
#plt.show()
    

index_file = open( sys.argv[1] + "/check_index.csv", "r")
output_file = open( sys.argv[2], "w+")

output_file.write("ID,Ans")
#remove title
index = index_file.readlines()[1::] 
for i in range(len(index)):
    temp = list(map(int, index[i].split(",")))
    output_file.write("\n" + str(i) + ",")
    if ( kmeans.labels_[ temp[1] ] == kmeans.labels_[ temp[2] ] ):
        output_file.write( str(1) )
    else:
        output_file.write( str(0) )

index_file.close()
output_file.close()

print("--- %s seconds ---" % (time.time() - start_time))
