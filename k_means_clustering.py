#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering

# ## Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os


# ## Importing the dataset


path_to = 'ML/files/'

names = 'qaccver saccver pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovhsp'
names = names.split(" ")

dataset = pd.read_csv('{}H-O_all.fa.csv'.format(path_to), names=names, sep=',')

ids = dataset.groupby('saccver')
#print(dataset)
count = ids["saccver"].value_counts()

ids_list = []
ids_100 = []

for i in ids:
    ids_list.append(i[0])

for i in ids_list:
    #print(i)
    if int(count[i]) > 100:
        ids_100.append(i) 
        
dataset_filt = pd.DataFrame()
dataset_filt['qaccver'] = dataset['qaccver'].loc[dataset['saccver'].isin(ids_100)]
dataset_filt['saccver'] = dataset['saccver'].loc[dataset['saccver'].isin(ids_100)]
dataset_filt['pident'] = dataset['pident'].loc[dataset['saccver'].isin(ids_100)]

dataset_filt['length'] = dataset['length'].loc[dataset['saccver'].isin(ids_100)]
dataset_filt['qcovhsp'] = dataset['qcovhsp'].loc[dataset['saccver'].isin(ids_100)]
#dataset_filt['ssciname'] = dataset['ssciname'].loc[dataset['saccver'].isin(ids_100)]

def clustering(data):
    
    comb = pd.DataFrame()
    dfs = []
    for i in ids_100:
        
        dfs.append(comb)
        ds = pd.DataFrame()
        
        ds = data.loc[data['saccver'] == i]
        
        #print(ds)
        X = ds.iloc[:, 2:-1].values
        print(X)
        from sklearn.cluster import KMeans
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i+1, init = 'k-means++', random_state = 0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            #print(wcss)
        x = range(1, len(wcss)+1)

        from kneed import KneeLocator
        kn = KneeLocator(x, wcss, curve='convex', direction='decreasing')
        elbow = kn.knee

        kmeans = KMeans(n_clusters = elbow+1, init = 'k-means++', random_state = 0)
        y_kmeans = kmeans.fit_predict(X)
        ds['cluster'] = y_kmeans
       # print(ds)
        #break
        dfs.append(ds)
        alldfs = pd.concat(dfs)
        
        
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
        plt.title('Clusters of IDs')
        plt.xlabel('%cov')
        plt.ylabel('%ID')
        plt.legend()
        plt.show()

    #print(alldfs)
    alldfs.to_csv('{}/cluster_out.csv'.format(path_to))
    
clustering(dataset_filt)
        

