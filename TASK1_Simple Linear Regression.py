#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('c:\\users\\joan\\ML_ZUMMIT_AFRICA TASKS\\archive-1\\nigeria_houses_data.csv')  #reading in the datset


# In[3]:


df.head()  #taking a look at the data


# In[4]:


df.describe()  #summarizing the data


# In[5]:


#there are only a few features with numerical values, so i will be using all instead of selecting a few

df1 = df[['bedrooms', 'bathrooms', 'toilets', 'parking_space', 'price']]

df1.head(9)


# In[6]:


#plotting each of the features above
graphs = df1[['bedrooms', 'bathrooms', 'toilets', 'parking_space', 'price']]

graphs.hist()

plt.show()


# In[7]:


#plotting each of the features against the dependent variable 'price' to see their linear relationship

plt.scatter(df1.bedrooms, df1.price, color = 'green')

plt.xlabel("bedrooms")

plt.ylabel("price")

plt.show()


# In[8]:


plt.scatter(df1.bathrooms, df1.price, color = 'green')

plt.xlabel("bathrooms")

plt.ylabel("price")

plt.show()


# In[9]:


plt.scatter(df1.toilets, df1.price, color = 'green')

plt.xlabel("toilets")

plt.ylabel("price")

plt.show()


# In[10]:


plt.scatter(df1.parking_space, df1.price, color = 'green')

plt.xlabel("parking_space")

plt.ylabel("price")

plt.show()


# I have noticed that in each scatter plot, there are outliers

# ### Creating train and test datset

# In[11]:


mask = np.random.rand(len(df)) < 0.8 

train = df1[mask]

test = df1[~mask]


# In[12]:


#training data distribution

plt.scatter(train.bedrooms, train.price, color ='green')

plt.xlabel("bedrooms")

plt.ylabel("bathrooms")

plt.show()


# In[13]:


# modeling the data

from sklearn import linear_model 

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['bedrooms']])

train_y = np.asanyarray(train[['price']])

regr.fit(train_x, train_y)

print("Coefficients: ", regr.coef_)

print("Intercept: ", regr.intercept_)


# In[14]:


#plotting the fit line over the data

plt.scatter(train.bedrooms, train.price, color = 'green')

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')


plt.xlabel("bedrooms")

plt.ylabel("price")


# In[15]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['bedrooms']])

test_y = np.asanyarray(test[['price']])

test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )


# ### selecting another feature to train with to compare the MAE

# In[16]:


# using the 'parkn\ing_space feature to train the data'

train_x = train[['parking_space']]

test_x = test[['parking_space']]


# In[17]:


regr.fit(train_x, train_y)


# In[18]:


prediction = regr.predict(test_x)


# In[19]:


#using the predictions and test_y data to find the mean absolute error
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(prediction - test_y)))


# There is only a slight difference  in the MAE when I trained using 'parking_space'

# #### Taking another feature as my independent variable, so i can compare the Mean absolute error and R^2 score

# In[20]:


# since the data has already neen split into training and test set, ill go right into training the data distribution and modelling...

plt.scatter(train.parking_space, train. price, color = 'yellow')
plt.xlabel("parking_space")
plt.ylabel("price")
plt.show()


# In[21]:


#modelling the data

train_x = np.asanyarray(train[["parking_space"]])
train_y = np.asanyarray(train[["price"]])
regr.fit(train_x, train_y)

print("Coefficents2: ", regr.coef_)
print("Intercept2: ", regr.intercept_)


# In[22]:


#plotting the fit line over the data

plt.scatter(train.parking_space, train.price, color = 'yellow')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '--b')
plt.xlabel("parking_space")
plt.ylabel("price")


# In[23]:


#calculating the MSE and R^2 score values:

test_x2 = np.asanyarray(test[['parking_space']])
test_y2 = np.asanyarray(test[['price']])
y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y2)))

print("Residual error: %.2f" % np.mean((y_hat - test_y2) ** 2))

print("R2_score: %.2f" % r2_score(test_y2, y_hat))


# Observations:
# 
# * Mean Absolute error is the same when i use parking_space and bathrooms as my x variable
# * The R2 score value is also the same when i use parking_space and bedroom as my x variable
# * the r2 score is a negative value, -0.05 which is 1.05 units away from 1. the r2 score value indicates lesser predictive accuracy for the dependent variable

# In[ ]:




