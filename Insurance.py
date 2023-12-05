import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(r'D:\machinelearning\100days\kmeans')
df=pd.read_csv(r'student_clustering.csv')
x=df.iloc[:,:].values

class k_means():
    def __init__(self,n_cluster,n_iteration):
        self.cluster=n_cluster
        self.iteration=n_iteration
        self.Centroid=None
    def fit_predict(self,x):
        k=random.sample(range(x.shape[0]),self.cluster)
        x=x
        self.Centroid=x[k]
        for i in range(self.iteration):
            cluster_groups=self.assign_cluster(x)
            old_centroid=self.Centroid
            self.Centroid=self.move_centroid(x,cluster_groups)
        return cluster_groups
    def assign_cluster(self,x):
        cluster_group=[]
        distance=[]
        for row in x:
            for centroids in self.Centroid:
                distance.append(np.sqrt(np.dot(row-centroids,row-centroids)))                
            min_distance=np.min(distance)
            index_pos=distance.index(min_distance)
            cluster_group.append(index_pos)
            distance=[]
        return np.array(cluster_group)
    def move_centroid(self,x,cluster_groups):        
        new_centroid=[]
        cluster_type=np.unique(cluster_groups)
        for type in cluster_type:
            new_centroid.append(x[cluster_groups==type].mean(axis=0))
        return np.array(new_centroid)
