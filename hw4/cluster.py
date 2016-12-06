#!/usr/bin/env python
# coding=utf-8
##############################################################
 # File Name : cluster.py
 # Purpose : clustering the title on the Stack OverFlow
 # Creation Date : Thu 24 Nov 2016 01:48:34 PM CST
 # Last Modified : Tue 06 Dec 2016 07:55:14 PM CST
 # Created By : SL Chung
##############################################################

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE

start_time = time.time()

#Set the tf-idf vectorizer
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2),
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

svd2 = TruncatedSVD(3)
visual = make_pipeline(svd2, normalizer)

#Array of the vectorized titles.
features = lsa.fit_transform(X) 
pic = visual.fit_transform(X) 

print("Features are total:", features.shape )

n = 120
kmeans = KMeans(n_clusters=n , random_state=0).fit(features)
    

print("Number in each cluster:", np.sort(np.bincount(kmeans.labels_) ))
'''
l = kmeans.labels_
for i in range(n):
    #index of same cluster
    index = ((l==0)*1).nonzero()[0]
    v = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    temp_corpus = [corpus[i] for i in index]
    v.fit_transform(temp_corpus)
    print("The", str(i), "cluster:", v.get_feature_names()[np.argmin(v.idf_)], index.shape)
    l -= 1

'''
pic = np.transpose(pic)
fig1 = plt.figure(1)
#ax1 = fig1.add_subplot(211, projection='3d')
test_color = kmeans.labels_
for i in range(n):
    #index of same cluster
    index = ((test_color==0)*1).nonzero()
    c = cm.hot(i / float(n))
    plt.scatter(pic[1][index], pic[2][index], color=c)
    #ax1.scatter(pic[0][index], pic[1][index], pic[2][index],c=c)
    test_color = test_color - 1
'''
real_label = np.load( sys.argv[1] + "/real_label.npy")
real_color = real_label
fig2 = plt.figure(2)
for i in range(20):
    #index of same cluster
    index = ((real_color==0)*1).nonzero()
    c = cm.hot(i / float(20))
    plt.scatter(pic[1][index], pic[2][index], color=c)
    real_color = real_color - 1
'''
plt.show()

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
