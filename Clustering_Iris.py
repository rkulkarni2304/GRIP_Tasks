#!/usr/bin/env python
# coding: utf-8

# # Predicting Optimum Clusters For Iris Dataset

# ## Author: Rahul Kulkarni

# ### Problem Statement:
# Predicting the number of clusters in the dataset using various clustering models.

# ### Importing Libraries 

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


iris_df = pd.read_csv('D:\GRIP_Tasks\Iris.csv')
features = iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
iris_df.head()


# ### Clustering Models

# There are various models that we can use, but we will test using the following algorithms:
# 1. K-means

# #### K-Means

# In[3]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[4]:


features_std = StandardScaler().fit_transform(features)
features_std


# In[5]:


wcss = []
for i in range(1,10):
    k_means = KMeans(init = "k-means++", n_clusters = i, n_init = 10)
    k_means.fit(features_std)
    wcss.append(k_means.inertia_)
plt.plot(range(1, 10), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# From the above plot we can clearly see that the optimum number of clusters is 3. It is pretty clear that increasing the clusters will cause decrease in the WCSS value. But the optimum solution lies at a point where the gradient of the curve suddenly decreases and is known as the elbow point. For above plot elbow point is 3 clusters.

# Let's create a k-means model with number of clusters equal to 3.

# In[7]:


k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 10)
k_means.fit(features_std)
species_predict = k_means.predict(features_std)


# Now let's visualize the clusters on a plot

# In[21]:


plt.figure(figsize=(9,7))
plt.scatter(features_std[species_predict == 0, 0], features_std[species_predict == 0, 1], 
            s = 50, c = 'red', label = 'Iris-setosa')
plt.scatter(features_std[species_predict == 1, 0], features_std[species_predict == 1, 1], 
            s = 50, c = 'blue', label = 'Iris-versicolour')
plt.scatter(features_std[species_predict == 2, 0], features_std[species_predict == 2, 1],
            s = 50, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:,1], 
            s = 50, c = 'yellow', label = 'Centroids')

plt.legend()


# ### Conclusion

# After calculating metrics such as WCSS it is clear that optimum number of clusters for the data set is 3.
