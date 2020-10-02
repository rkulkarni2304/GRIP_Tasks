#!/usr/bin/env python
# coding: utf-8

# # Predctiction Of Percentage Of Marks For Students
# 

# ### Author: Rahul Kulkarni
# 

# ### Problem Statement:
# Create a regression model for predicting marks for students.

# ### Importing required libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ### Exploratory Data Analysis

# In[2]:


students_df = pd.read_csv("http://bit.ly/w-data")
students_df.head()


# Therefore we are provided with 2 variables: Hours studied by a student per day and percentage of score recieved by the student.

# Lets visualize the data on a regression plot to get a better understanding of the relation between the variables

# In[4]:


hours = students_df['Hours']
score = students_df['Scores']
sns.regplot(x=hours,y=score)
plt.show()


# It is clearly visible that there is a linear variation between Hours studied by a student and Score. Furthermore it is a linearly increasing variation.

# Lets check to what degree does the independant variable('Hours') affect the target variable('Score')

# In[5]:


corr_matrix = students_df.corr()
print(corr_matrix)
sns.heatmap(corr_matrix)
plt.show()


# As we can see from the correlation coefficient values, 'Hours' and 'Score' are strongly related.

# ### Modelling

# As the data is in the form of continuos variables the best model to use for prediction is the regression model.From the scatter plot and correlation values it is evident that simple linear regression would give an accurate prediction.

# ### Importing libraries for modelling

# In[6]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[7]:


lr = LinearRegression()
hours = np.array(hours).reshape(-1,1)
score = np.array(score)
lr.fit(hours,score)
score_predict = lr.predict(hours)
score_predict


# Let's see the equation for the regression line generated by the model.

# In[8]:


coefficient = lr.coef_
intercept = lr.intercept_
equation='Score = '+str(coefficient[0])+'*Hours + '+str(intercept)
equation


# ### Model Analysis

# To check how accurate our model is, we can calculate various statistical metrics.

# 1. Cofficient of determination:

# In[9]:


lr.score(hours,score)


# The value for coefficient of determination is high which means that 95.29% change is observed in the score due to hours studied by the student.

# 2. Mean Squared Error

# In[10]:


mean_squared_error(score,score_predict)


# The value of mean squared error is low hence the model seems to be a good fit.

# 3. Mean Absolute Error

# In[11]:


mean_absolute_error(score,score_predict)


# The value of mean absolute error is also low hence the model seems to be a good fit.

# Let's take a look at the distribution plot between the actual score values and the predicted score values.

# In[12]:


#distribution plot
ax1 = sns.distplot(score,hist=False,color='r',label='Actual Value')
sns.distplot(score_predict,hist=False,color='b',label='Fitted Values', ax=ax1)
plt.show()


# The above plot shows clearly how our predicted values almost overlap the given score values.

# ### Predicting Marks for Test Data

# Let's predict the score for a student who studied for 9.25 hours/day.

# In[13]:


lr.predict([[9.25]])


# Therefore our prediction for hours = 9.25 is a score of 92.91%. Which seems to be quite accurate since the study hour is very high.

# ### Conclusion

# We have predicted the marks for students using a linear regression model which seems to be a fairly accurate one, which can also be seen from the various statistical metrics calculated.