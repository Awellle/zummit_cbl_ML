#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


books_df= pd.read_csv('C:\\Users\\JOAN\\ML_ZUMMIT_AFRICA TASKS\\archive\Books.csv')


# In[3]:


books_df.head()


# In[4]:


books_df = books_df.drop_duplicates()


# In[6]:


books_df.info()


# In[7]:


books_df.shape


# In[8]:


ratings_df = pd.read_csv('C:\\Users\\JOAN\ML_ZUMMIT_AFRICA TASKS\\archive\\Ratings.csv')


# In[9]:


ratings_df.head()


# In[10]:


ratings_df.info()


# In[11]:


#adding user input
user_input = [
    {'Book-Title': "Where You'll Find Me: And Other Stories", 'Book-Rating':5},
    {'Book-Title': "The Kitchen God's Wife", 'Book-Rating':0},
    {'Book-Title': "Nights Below Station Street", 'Book-Rating':0},
    {'Book-Title': "The Witchfinder (Amos Walker Mystery Series)", 'Book-Rating':6},
    {'Book-Title': "Jane Doe", 'Book-Rating':5},
    
]

inputBooks = pd.DataFrame(user_input)
inputBooks


# In[14]:


#Filtering out the books by ISBN
input_id= books_df[books_df['Book-Title'].isin(inputBooks['Book-Title'].tolist())]


# In[15]:


input_id


# In[21]:


#Then merging it so I can get the ISBN. 
inputBooks = pd.merge(input_id, inputBooks)
inputBooks


# In[28]:


#inputBooks = inputBooks.drop('Year-Of-Publication,1).drop('Image-URL-S',1).drop('Image-URL-M',1).drop('Image-URL-L',1).drop('Publisher',1)
#inputBooks


# In[27]:


inputBooks = inputBooks.drop('Year-Of-Publication', 1)
inputBooks


# In[29]:


inputBooks = inputBooks.drop('Publisher', 1)
inputBooks


# In[30]:


inputBooks = inputBooks.drop('Image-URL-S', 1)
inputBooks


# In[31]:


inputBooks = inputBooks.drop('Image-URL-M', 1)
inputBooks


# In[32]:


inputBooks = inputBooks.drop('Image-URL-L', 1)
inputBooks


# In[33]:


inputBooks = inputBooks.drop('Book-Author', 1)
inputBooks


# In[34]:


#getting users who have read the same books that the input has read
user_subset = ratings_df[ratings_df['ISBN'].isin(inputBooks['ISBN'].to_list())]
user_subset


# In[36]:


#groupong the rows by 'ISBN'
user_subset_group = user_subset.groupby(['ISBN'])
user_subset_group.head()


# In[37]:


#sorting the user subset group so that users that have read the same books will have higher priority

user_subset_group = sorted(user_subset_group,  key=lambda x: len(x[1]), reverse=True)
user_subset_group[0:3]


# In[39]:


#selecting a small part of the user subset group on which Peasrons Correlation function will be calculated on
user_subset_group = user_subset_group[0:70]


# In[43]:


pearsonCorrelationDict = {} #initializing an empty dictionary to store the pearson coefficients

#For every user group in our subset
for name, group in user_subset_group: #start a loop over the subset of users(user_subset_group). Each user's data is representaed as a group and name represnts the user's identifier
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='ISBN') #sorts the data of the current user group by isbn
    inputBooks = inputBooks.sort_values(by='ISBN') #sorts the movie rated by the input user by isbn
    #Get the N for the formula. This line indicates that the next line calculates the number of common books rated by both users
    nRatings = len(group) # sets nRatings to the number of ratings in the curret user group 
    #Get the review scores for the movies that they both have in common
    temp_df = inputBooks[inputBooks['ISBN'].isin(group['ISBN'].tolist())] #creates a temporary DF of books from inputBooks that are also rated by the current user in the loop.
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['Book-Rating'].tolist() #converts the ratings of the common books from the input user group into a list.
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['Book-Rating'].tolist() # converts the ratings of the same books from the current user group into another list
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings) # calculates the sum of squares of the differences from the mean for ratings by the input user
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings) # calculates the sum of squares of the differences from the mean for ratings by the current user.
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings) # calculates the sum of the product of paired ratings minus the product of the sum of ratings divided by the number of ratings
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0: # checks if neither of the denominator components is zero to avoid division by zero
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy) #calculates the pearson correlation coefficient if the condition is met and stores it in the dictionary under the user's name
    else: #handles the case where the denominator might be zero
        pearsonCorrelationDict[name] = 0 #assigns a correlation of 0 if the calculation cannot be performed due to a zero denominator


# In[45]:


pearsonCorrelationDict.items()


# In[46]:


pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index') #creates a pandas DataFrame from the dictionary, pearsonCorrelationDict, where each key-value pair in the dictionary becomes a row in the dataframe. The dict keys are used as the index of the dataframe, and the values are placed in the DataFrame's columns. the orient = 'index' parameter specifies that the dictionary keys should be used as row labels 


# In[48]:


pearsonDF.head()


# In[49]:


pearsonDF.columns = ['similarityIndex'] # renames the column in the dataframe to 'similarityIndex'. initially, the DF has only one column i.e the pearson correlation values, and this command assigns a more descriptive name to that column


# In[50]:


pearsonDF.head()


# In[51]:


pearsonDF['ISBN'] = pearsonDF.index # creates a new column called ISBN, which is populated with the index of the dataFrame. This index contains the ISBN from the dict


# In[52]:


pearsonDF.head()


# In[53]:


pearsonDF.index = range(len(pearsonDF)) # resets the index of the DF to a simple ascending sequence of integers, starting from 0 up to the length of the DF minus one. useful for making the DF easier to work with, esp. if the original index (user IDs) will be used as a regular column
pearsonDF.head()


# In[57]:


#30 users that are similar to the input
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=True)[0:30]
topUsers.head()


# In[58]:


topUsersRating=topUsers.merge(ratings_df, left_on='ISBN', right_on='ISBN', how='inner') #left_on = userId and right_on = userId specify the columns in each DataFrame (topUsers and ratings_df respectively) that pandas will use to perform the merge. 'inner' specifies the type of merge to perform. an inner merge means that the result will only include rows that have matching values in both DataFrames. its similar to an inner join in SQL.
topUsersRating.head()


# In[60]:


#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['Book-Rating']
topUsersRating.head()


# In[61]:


tempTopUsersRating = topUsersRating.groupby('ISBN').sum()[['similarityIndex','weightedRating']]


# In[62]:


tempTopUsersRating


# In[64]:


tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head(10)


# In[65]:


recommendation_df = pd.DataFrame()  #CREATING AN EMPTY DATAFRAME


# In[66]:


recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df


# In[67]:


recommendation_df['ISBN'] = tempTopUsersRating.index
recommendation_df


# In[68]:


recommendation_df.head()


# In[69]:


recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)  #sorting the dataframe 'recommendation_df'


# In[70]:


recommendation_df.head(10)  #checking to see the 10 books that the algorithm recommended


# In[72]:


books_df.loc[books_df['ISBN'].isin(recommendation_df.head(10)['ISBN'].tolist())]


# In[ ]:




