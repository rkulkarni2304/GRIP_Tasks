#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classifier For Iris Dataset

# ## Author : Rahul Kulkarni

# ### Problem Statement: 
# Create a decision tree classifier for the dataset and visualize it graphically

# ### Importing required libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ### Exploratory Data Analysis

# In[10]:


iris_df = pd.read_csv('D:\GRIP_Tasks\Iris.csv')
iris_df.head()


# This is a classification problem since the target variable is of categorical type. We have 4 independant variables Sepal Length, Sepal Width, Petal Length and Petal Width all of which have the cm unit. The target variable is Species which falls into 3 categories Iris_setosa, Iris-versicolor and Iris-virginica.

# Let's group the data by the species and have a look at various mathematical attributes like mean, mode, range etc.

# In[14]:


iris_df.groupby('Species').describe()


# Each of the species have equal number of data points (50) and we can see an increasing trend of their means. Let's take a better look at the data with the help of a boxplot.

# In[18]:


features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
for feature in features:
    sns.boxplot(x='Species',y=feature,data = iris_df)
    plt.show()


# From the above box plots the relation between:
# 1. SepalLength and Species: It is a linearly increasing relation.
# 2. SepalWitdth and Species: It looks like an exponential relation due to the U- shaped curve formed.
# 3. PetalLength,PetalWidth and Species: It is a linearly incresing relation with higher slope.

# Let's take a look at the distribution plots to find any overlaps between variables.

# In[31]:


for feature in features:
    sns.displot(x=feature,data=iris_df,hue='Species',kind='kde')
    plt.show()


# As we can see from the distribution plot of SepalWidth there is high overlap between the species and from the SepalLength plot we can observe a slight overlap. But from the petal features plot we can observe that there is an slight overlap between Iris-versicolor and Iris-virginica but none of them overlap with Iris-setosa.

# ### Modelling

# It's clear that this is a classification modelling problem and we will use a decision tree classifier to predict the species.

# In[32]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[43]:


features = np.array(iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']])
species = np.array(iris_df['Species'])
dtree = DecisionTreeClassifier(criterion='entropy',max_depth=5)
model = dtree.fit(features,species)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtree, 
                   feature_names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],  
                   class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                   filled=True)


# ### Model Analysis

# Let's analyse our model by checking the cross validation score for 5 folds.

# In[41]:


from sklearn.model_selection import cross_val_score
cross_val_score(dtree,features,species,cv=5)


# Our model seems to be a good fit since the accuracy scores for vairous test-train split are very high and close to 1.
