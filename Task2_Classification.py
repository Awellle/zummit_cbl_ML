#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('c:\\users\\joan\\OneDrive\\Desktop\\datasets for tasks\\archive-1\\creditcard_2023.csv')


# In[5]:


df.head()


# In[4]:


df['Class'].value_counts()


# 284315 people belong to class 0, and 284315 people belong to class 1

# In[8]:


X = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'Amount']].values
X[0:5]


# In[7]:


y = df['Class'].values
y[0:5]


# In[9]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))  #normalizing the data
X[0:5]


# In[10]:


from sklearn.model_selection import train_test_split  #splitting the data set into testing and training set. test_size = 0.3 indicates that 30% of the data will be used for testing whie the remaining 70% percent will be used for training the data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[11]:


# training the model 
from sklearn.neighbors import KNeighborsClassifier
k = 5  
neighbors = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neighbors


# In[13]:


y_hat = neighbors.predict(X_test)
y_hat[0:5]


# In[15]:


#evaluating the accuracy of my prediction

from sklearn import metrics
print("train set accuracy: ", metrics.accuracy_score(y_train, neighbors.predict(X_train)))
print("test set accuracy: ", metrics.accuracy_score(y_test, y_hat))


# The accuracies for both the train set and test set are high, indicating good performance. It also suggests that there is minimal overfitting.

# In[ ]:


#Ks = 10
#mean_acc = np.zeros((Ks-1))
#std_acc = np.zeros((Ks-1))

#for n in range(1,Ks):
    
    #Train Model and Predict  
    #neighbors = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    #y_hat=neighbors.predict(X_test)
    #mean_acc[n-1] = metrics.accuracy_score(y_test, y_hat)

    
    #std_acc[n-1]=np.std(y_hat==y_test)/np.sqrt(y_hat.shape[0])

#mean_acc


# In[1]:


#plt.plot(range(1,Ks),mean_acc,'g')
#plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
#plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
#plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
#plt.ylabel('Accuracy ')
#plt.xlabel('Number of Neighbors (K)')
#plt.tight_layout()
#plt.show()


# ### Using Decision trees 

# In[6]:


df.head()


# In[8]:


X1 = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'Amount']].values
X1[0:5] 


# In[9]:


y1 = df['Class']
y1[0:5]


# In[10]:


from sklearn.model_selection import train_test_split


# In[12]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 3)


# In[13]:


#checking for the size of the train and test test

print("X training set size: ", X1_train.shape)
print("Y training size: ", y1_train.shape)
print("X testing size: ", X1_test.shape)
print("Y testing size: ", y1_test.shape)


# After checking the shape, it is seen that the dimensions match. i can then go on with modeling

# In[14]:


#training

from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

cardTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
cardTree


# In[15]:


cardTree.fit(X1_train, y1_train)


# In[16]:


prediction= cardTree.predict(X1_test)  #making a prediction


# In[17]:


#comparing the actual values with predicted values
print(y1_test[0:5])
print(prediction[0:5])


# In[18]:


#checking the accuracy of the model 
from sklearn import metrics

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y1_test, prediction))


# ### Using Logistic Regression

# In[19]:


import pylab as pl
import scipy.optimize as opt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


df.head()


# In[23]:


print(df['Class'].dtype)


# In[28]:


#selecting features to be used for modeling
#the dependendent variable is already of type int 

#df = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'Amount', 'Class']]
#df.head()


# In[29]:


#df1 = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'Amount', 'Class']]
#df1.head()


# In[30]:


print(df.columns)


# In[33]:


df = pd.read_csv('c:\\users\\joan\\OneDrive\\Desktop\\datasets for tasks\\archive-1\\creditcard_2023.csv')


# In[34]:


df = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'Amount', 'Class']]
df.head()


# In[35]:


#defining X and y

X2 = np.asarray(df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'Amount']])
X2[0:5]

y2 =np.asarray(df['Class'])
y2[0:5]


# In[38]:


#normalizing the data 
from sklearn import preprocessing

X2= preprocessing.StandardScaler().fit(X2).transform(X2)
X2[0:5]


# In[39]:


#getting the training set and testing set

X2_train, X2_test, y2_train, y2_test = train_test_split( X2, y2, test_size=0.2, random_state=4)
print ('Train set:', X2_train.shape,  y2_train.shape)
print ('Test set:', X2_test.shape,  y2_test.shape)


# In[41]:


#fitting the data to the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X2_train,y2_train)
LR


# In[42]:


#making a prediction using the test set
y_ = LR.predict(X2_test)
y_


# In[43]:


y_prob = LR.predict_proba(X2_test)
y_prob


# In[44]:


#checking the accuracy of the model using jaccard score
from sklearn.metrics import jaccard_score
jaccard_score(y2_test, y_,pos_label=0)


# A jaccard score of 88% indicates closeness in accuracy of the predicted values and actual values

# ### Using SVM

# In[1]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


credit_df = pd.read_csv('c:\\users\\joan\\OneDrive\\Desktop\\datasets for tasks\\archive-1\\creditcard_2023.csv')


# In[3]:


credit_df.head()


# In[4]:


# all features/variables contain numerical data


# In[4]:


X = np.asarray(credit_df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'Amount']])
X[0:5]


# In[5]:


y = np.asarray(credit_df['Class'])
y[0:5]


# In[6]:


#splitting the data into train and test sets and checking their size
X3_train, X3_test, y3_train, y3_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X3_train.shape,  y3_train.shape)
print ('Test set:', X3_test.shape,  y3_test.shape)


# ### modeling using the Radial Bias Function
# 
# 

# In[7]:


from sklearn import svm


# In[8]:


model = svm.SVC(kernel='rbf')


# model.fit(X3_train, y3_train)    # i kept trying to fit, to no avail. 

# In[ ]:


y__ = model.predict(X3_test)
y__[0:5]

