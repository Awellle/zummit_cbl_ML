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


# selecting another feature to train with to compare the MAE¶

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


# ### Observation
# * There is only a slight difference in the MAE when I trained using 'parking_space'

# ## Taking another feature as my independent variable, so i can compare the Mean absolute error and R^2 score

# In[20]:


# since the data has already neen split into training and test set, ill go right into training the data distribution and modelling...

plt.scatter(train.parking_space, train. price, color = 'yellow')
plt.xlabel("parking_space")
plt.ylabel("price")
plt.show()


# In[21]:


#modelling the data

train_x1 = np.asanyarray(train[["parking_space"]])
train_y1 = np.asanyarray(train[["price"]])
regr.fit(train_x1, train_y1)

print("Coefficents1: ", regr.coef_)
print("Intercept1: ", regr.intercept_)


# In[22]:


#plotting the fit line over the data

plt.scatter(train.parking_space, train.price, color = 'yellow')
plt.plot(train_x1, regr.coef_[0][0]*train_x1 + regr.intercept_[0], '--b')
plt.xlabel("parking_space")
plt.ylabel("price")


# In[23]:


#calculating the MSE and R^2 score values:

test_x1 = np.asanyarray(test[['parking_space']])
test_y1 = np.asanyarray(test[['price']])
y_hat = regr.predict(test_x1)

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y1)))

print("Residual error: %.2f" % np.mean((y_hat - test_y1) ** 2))

print("R2_score: %.2f" % r2_score(test_y1, y_hat))


# Observations:
# 
# * Mean Absolute error is the same when i use parking_space and bathrooms as my x variable
# * The R2 score value is also the same when i use parking_space and bedroom as my x variable
# * the r2 score is a negative value, -0.04 which is 1.04 units away from 1. the r2 score value indicates lesser predictive accuracy for the dependent variable

# ## Multiple Linear regression model

# In[25]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['bedrooms','bathrooms','toilets',]])
y = np.asanyarray(train[['price']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)


# In[26]:


x = np.asanyarray(test[['bedrooms','bathrooms','toilets']])
y = np.asanyarray(test[['price']])
y_hat= regr.predict(test[['bedrooms','bathrooms','toilets']])

print("Mean Squared Error (MSE) : %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# #### Observation
# * a variance score of 1 indicates better accuracy. here, the variance score (-0.04) is really low, indicating that the accuracy of the model is low
# * 
# * i will then change one variable in my train set and then check to see if the prediction would be better

# In[30]:


x1 = np.asanyarray(train[['bedrooms','bathrooms','parking_space']])
y1 = np.asanyarray(train[['price']])
regr.fit(x1, y1)
print("Coefficients: ", regr.coef_)

x1 = np.asanyarray(test[['bedrooms','bathrooms','parking_space']])
y1 = np.asanyarray(test[['price']])
y_= regr.predict(test[['bedrooms','bathrooms','parking_space']])

print("MSE: %.2f" % np.mean((y_ - y1) **2))

print("Variance score: %.2f" % regr.score(x1, y1))


# ### Observation
# 
# * the new variance score (-0.06) appears to be worse than the first
# * now, i will try again, using all the variables in the dataset

# In[31]:


x2 = np.asanyarray(train[['bedrooms','bathrooms','parking_space', 'toilets']])
y2 = np.asanyarray(train[['price']])
regr.fit(x2, y2)
print("Coefficients: ", regr.coef_)

x2 = np.asanyarray(test[['bedrooms','bathrooms','parking_space', 'toilets']])
y2 = np.asanyarray(test[['price']])
y__= regr.predict(test[['bedrooms','bathrooms','parking_space', 'toilets']])

print("MSE: %.2f" % np.mean((y__ - y2) **2))

print("Variance score: %.2f" % regr.score(x2, y2))


# * THERE IS NO CHANGE IN THE VARIANCE SCORE
# * Hence, the first multiple linear regression model is the best thus far
# * since the trend of the data is not curvy, i cannot attempt to use a polynomial regression model, so i will stop here. 

# In[ ]:




