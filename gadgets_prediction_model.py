#!/usr/bin/env python
# coding: utf-8

# ## MLR(1)

# In[58]:


import pandas as pd
data_= pd.read_csv('C:\\Users\\JOAN\\group1_task_zummit\\electronics_products_pricing.csv') #reading in the data


# In[59]:


data_.isnull().sum()


# In[60]:


data_ = data_.dropna()


# In[61]:


data_.isnull().sum()


# In[62]:


data_.columns


# In[63]:


data_.head()


# In[64]:


# removing columns
removed = ['id', 'prices.shipping', 'brand', 'manufacturer', 'manufacturerNumber', 'name', 'ean', 'upc', 'dateUpdated', 'categories', 'prices.merchant', 'prices.currency', 'prices.dateSeen', 'dateAdded', 'manufacturerNumber', 'asins', 'imageURLs', 'prices.sourceURLs', 'keys', 'sourceURLs']
data_ = data_.drop(columns=removed)


# In[65]:


data_.head()


# In[66]:


data_['prices.isSale'] = data_['prices.isSale'].astype(int)


# In[67]:


data_.head()


# In[68]:


# convert weight
import re
def convert_to_kg(weight):
    if pd.isna(weight):
        return None
    # Convert to lower case and strip any whitespace
    weight = weight.lower().strip()
    
    # Check for different units and convert to kg
    if 'pounds' in weight or 'pound' in weight or 'lb' in weight:
        # Extract numeric value and convert to kg
        pounds = float(re.findall(r'\d*\.?\d+', weight)[0])
        return pounds * 0.453592
    elif 'ounces' in weight or 'ounce' in weight or 'oz' in weight:
        # Extract numeric value and convert to kg
        ounces = float(re.findall(r'\d*\.?\d+', weight)[0])
        return ounces * 0.0283495
    elif 'kg' in weight:
        # Extract numeric value (already in kg)
        return float(re.findall(r'\d*\.?\d+', weight)[0])
    elif 'g' in weight:
        # Extract numeric value and convert to kg
        grams = float(re.findall(r'\d*\.?\d+', weight)[0])
        return grams / 1000.0
    else:
        return None


# In[69]:


data_['weight'] = data_['weight'].apply(convert_to_kg)


# In[70]:


data_.head()


# In[71]:


data_ = pd.get_dummies(data_)


# In[72]:


data_.head()


# In[73]:


data_.isnull().sum()


# In[74]:


data_ = data_.dropna()


# In[75]:


import numpy as np
print("Infinite values:\n", np.isinf(data_).sum())


# In[76]:


# Normalize the price
data_['price'] = pd.to_numeric(data_['price'], errors='coerce')
data_= data_.dropna(subset=['price'])


# In[77]:


#removing the outliers in the price column
#calculating the interquartile range for the 'price' column
Q1 = data_['price'].quantile(0.25)
Q3 = data_['price'].quantile(0.75)

IQR = Q3 - Q1

#defining the acceptable range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#filtering the data to remove outliers
data_filtered = data_[(data_['price'] >= lower_bound) & (data_['price'] <= upper_bound)]

print("after filtering: ")
print(data_filtered['price'].describe())


# In[78]:


##EDA
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (10,5))
plt.scatter(data_filtered.index, data_filtered['price']) #a seaborn scatter plot to visualize weight against data
plt.title('Scatter plot of Price')
plt.xlabel('index')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[79]:


data_.head()


# In[80]:


#correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix = data_.corr()
plt.figure(figsize = (15,15))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm')
plt.show()


# In[81]:


#printing correlation variables with the target variable
print(correlation_matrix['price'].sort_values(ascending = False))


# In[82]:


data_.isnull().sum()


# In[83]:


##EDA
#analysing the relationship between price and weight

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'weight', y = 'price', data = data_) #a seaborn scatter plot to visualize weight against data
plt.title('weight vs. Price')
plt.xlabel('weight')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[84]:


#removing the outliers in the weight column
#calculating the interquartile range for the 'price' column
Q1 = data_['weight'].quantile(0.25)
Q3 = data_['weight'].quantile(0.75)

IQR = Q3 - Q1

#defining the acceptable range
lower_bound = Q1 - 1.0 * IQR
upper_bound = Q3 + 1.0 * IQR

#filtering the data to remove outliers
data_filtered = data_[(data_['weight'] >= lower_bound) & (data_['weight'] <= upper_bound)]

print("after filtering: ")
print(data_filtered['weight'].describe())


# In[106]:


##EDA
#analysing the relationship between price and weight

#plt.figure(figsize = (10,5))
#sns.scatterplot(x= 'weight', y = 'price', data = data_filtered) #a seaborn scatter plot to visualize weight against data
#plt.title('weight vs. Price')
#plt.xlabel('weight')  # sets the x label of the plot to be 'weight'
#plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
#plt.show()


# In[86]:


##EDA
#analysing the relationship between price and prices.condition_New

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'prices.condition_New', y = 'price', data = data_) #a seaborn scatter plot to visualize weight against data
plt.title('prices.condition_New vs. Price')
plt.xlabel('prices.condition_New')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[87]:


#removing the outliers in the prices.condition_New column
#calculating the interquartile range for the 'prices.condition_New' column
Q1 = data_['prices.condition_New'].quantile(0.25)
Q3 = data_['prices.condition_New'].quantile(0.75)

IQR = Q3 - Q1

#defining the acceptable range
lower_bound = Q1 - 0.005 * IQR
upper_bound = Q3 + 0.005 * IQR

#filtering the data to remove outliers
data_filtered = data_[(data_['prices.condition_New'] >= lower_bound) & (data_['prices.condition_New'] <= upper_bound)]
print("after filtering: ")
print(data_filtered['prices.condition_New'].describe())


# In[107]:


##EDA
#analysing the relationship between price and prices.condition_New

#plt.figure(figsize = (10,5))
#sns.scatterplot(x = 'prices.condition_New', y = 'price', data = data_filtered) #a seaborn scatter plot to visualize weight against data
#plt.title('prices.condition_New vs. Price')
#plt.xlabel('prices.condition_New')  # sets the x label of the plot to be 'weight'
#plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
#plt.show()


# In[89]:


##EDA
#analysing the relationship between price and prices.availability_Special Order

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'prices.availability_Special Order', y = 'price', data = data_) #a seaborn scatter plot to visualize weight against data
plt.title('prices.availability_Special Order vs. Price')
plt.xlabel('prices.availability_Special Order')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[90]:


#removing the outliers in the prices.availability_Special Order column
#calculating the interquartile range for the 'prices.availability_Special Order' column
Q1 = data_['prices.availability_Special Order'].quantile(0.25)
Q3 = data_['prices.availability_Special Order'].quantile(0.75)

IQR = Q3 - Q1

#defining the acceptable range
lower_bound = Q1 - 0.05 * IQR
upper_bound = Q3 + 0.05 * IQR

#filtering the data to remove outliers
data_filtered = data_[(data_['prices.availability_Special Order'] >= lower_bound) & (data_['prices.availability_Special Order'] <= upper_bound)]
print("after filtering: ")
print(data_filtered['prices.availability_Special Order'].describe())


# In[108]:


##EDA
#analysing the relationship between price and prices.availability_Special Order

#plt.figure(figsize = (10,5))
#sns.scatterplot(x = 'prices.availability_Special Order', y = 'price', data = data_filtered) #a seaborn scatter plot to visualize weight against data
#plt.title('prices.availability_Special Order vs. Price')
#plt.xlabel('prices.availability_Special Order')  # sets the x label of the plot to be 'weight'
#plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
#plt.show()


# In[91]:


##EDA
#analysing the relationship between price and prices.availability_Yes

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'prices.availability_Yes', y = 'price', data = data_) #a seaborn scatter plot to visualize weight against data
plt.title('prices.availability_Yes vs. Price')
plt.xlabel('prices.availability_Yes')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[92]:


#removing the outliers in the prices.availability_Yes column
#calculating the interquartile range for the 'prices.condition_New' column
Q1 = data_['prices.availability_Yes'].quantile(0.25)
Q3 = data_['prices.availability_Yes'].quantile(0.75)

IQR = Q3 - Q1

#defining the acceptable range
lower_bound = Q1 - 0.25 * IQR
upper_bound = Q3 + 0.25 * IQR

#filtering the data to remove outliers
data_filtered = data_[(data_['prices.availability_Yes'] >= lower_bound) & (data_['prices.availability_Yes'] <= upper_bound)]
print("after filtering: ")
print(data_filtered['prices.availability_Yes'].describe())


# In[109]:


##EDA
#analysing the relationship between price and prices.availability_Yes

#plt.figure(figsize = (10,5))
#sns.scatterplot(x = 'prices.availability_Yes', y = 'price', data = data_filtered) #a seaborn scatter plot to visualize weight against data
#plt.title('prices.availability_Yes vs. Price')
#plt.xlabel('prices.availability_Yes')  # sets the x label of the plot to be 'weight'
#plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
#plt.show()


# In[94]:


##EDA
#analysing the relationship between price and primaryCategories_Electronics,Furniture 

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'primaryCategories_Electronics,Furniture', y = 'price', data = data_) #a seaborn scatter plot to visualize weight against data
plt.title('primaryCategories_Electronics,Furniture vs. Price')
plt.xlabel('primaryCategories_Electronics,Furniture')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[95]:


#removing the outliers in the primaryCategories_Electronics,Furniture column
#calculating the interquartile range for the 'primaryCategories_Electronics,Furniture' column
Q1 = data_['primaryCategories_Electronics,Furniture'].quantile(0.25)
Q3 = data_['primaryCategories_Electronics,Furniture'].quantile(0.75)

IQR = Q3 - Q1

#defining the acceptable range
lower_bound = Q1 - 1.0 * IQR
upper_bound = Q3 + 1.0 * IQR

#filtering the data to remove outliers
data_filtered = data_[(data_['primaryCategories_Electronics,Furniture'] >= lower_bound) & (data_['primaryCategories_Electronics,Furniture'] <= upper_bound)]


# In[110]:


#EDA
#analysing the relationship between price and prices.availability_Yes

#plt.figure(figsize = (10,5))
#sns.scatterplot(x = 'primaryCategories_Electronics,Furniture', y = 'price', data = data_filtered) #a seaborn scatter plot to visualize weight against data
#plt.title('primaryCategories_Electronics,Furniture vs. Price')
#plt.xlabel('primaryCategories_Electronics,Furniture')  # sets the x label of the plot to be 'weight'
#plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
#plt.show()


# In[96]:


##EDA
#analysing the relationship between price and prices.isSale

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'prices.isSale', y = 'price', data = data_) #a seaborn scatter plot to visualize weight against data
plt.title('prices.isSale vs. Price')
plt.xlabel('prices.isSale')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[97]:


#removing the outliers in the prices.availability_Yes column
#calculating the interquartile range for the 'prices.condition_New' column
Q1 = data_['prices.isSale'].quantile(0.25)
Q3 = data_['prices.isSale'].quantile(0.75)

IQR = Q3 - Q1

#defining the acceptable range
lower_bound = Q1 - 1.0 * IQR
upper_bound = Q3 + 1.0 * IQR

#filtering the data to remove outliers
data_filtered = data_[(data_['prices.isSale'] >= lower_bound) & (data_['prices.isSale'] <= upper_bound)]
print("after filtering: ")
print(data_filtered['prices.isSale'].describe())


# In[111]:


#EDA
#analysing the relationship between price and prices.availability_Yes

#plt.figure(figsize = (10,5))
#sns.scatterplot(x = 'prices.isSale', y = 'price', data = data_filtered) #a seaborn scatter plot to visualize weight against data
#plt.title('prices.isSale vs. Price')
#plt.xlabel('prices.isSale')  # sets the x label of the plot to be 'weight'
#plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
#plt.show()


# In[99]:


data_filtered.head()


# In[101]:


# Split the data into features and target
X = data_filtered[['prices.isSale', 'prices.condition_New', 'prices.availability_Special Order', 'prices.availability_Yes', 'primaryCategories_Electronics,Furniture', 'prices.isSale']]
y = data_filtered['price']


# In[102]:


y.head()


# In[104]:


# Split the data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[105]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[106]:


y_pred = model.predict(X_test)


# In[107]:


mse = mean_squared_error(y_test, y_pred)
mse


# In[108]:


y_pred


# In[109]:


y_test


# In[110]:


# Calculate R-squared
r_squared = model.score(X_test, y_test)

# Display the results
mse, r_squared


# In[111]:


from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[115]:


# Feature Selection
selector = SelectKBest(score_func=f_regression, k=6)  # Select top 20 features
# Ridge Regression
ridge = Ridge()
# Pipeline
pipeline = Pipeline([('selector', selector), ('ridge', ridge)])


# In[116]:


# Hyperparameter Tuning
param_grid = {'ridge__alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')


# In[117]:


# Fit the model
grid_search.fit(X_train, y_train)


# In[118]:


# Make predictions
y_pred = grid_search.predict(X_test)


# In[119]:


r_squared = grid_search.best_estimator_.score(X_test, y_test)


# In[120]:


mse, r_squared, grid_search.best_params_


# In[121]:


# Apply log transformation to the target variable
y_log = np.log1p(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Train and evaluate the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Transform predictions back to original scale
y_pred_exp = np.expm1(y_pred)
y_test_exp = np.expm1(y_test)

# Evaluate the model
mse = mean_squared_error(y_test_exp, y_pred_exp)
r_squared = model.score(X_test, y_test)

mse, r_squared


# In[122]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor()

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

grid_search_gbr = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_gbr.fit(X_train, y_train)

# Make predictions
y_pred_gbr = grid_search_gbr.predict(X_test)

# Evaluate the model
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
r_squared_gbr = grid_search_gbr.best_estimator_.score(X_test, y_test)

mse_gbr, r_squared_gbr, grid_search_gbr.best_params_


# # RandomForest Regression
# 

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


# In[63]:


import pandas as pd
df = pd.read_csv('C:\\Users\\JOAN\\group1_task_zummit\\electronics_products_pricing.csv') #reading in the data


# In[64]:


df = df.dropna()


# In[65]:


df.head()


# In[66]:


df['primaryCategories'].unique


# In[67]:


df.columns


# In[68]:


# Convert categorical data to numerical data
df['prices.isSale'] = df['prices.isSale'].astype(int)
columns= ['prices.availability', 'primaryCategories', 'prices.condition']
df = pd.get_dummies(df, columns=columns)


# In[69]:


df.head()


# In[70]:


# removing columns
removed = ['id', 'ean', 'prices.shipping', 'upc', 'name', 'dateUpdated', 'brand', 'prices.merchant', 'prices.currency', 'prices.dateSeen', 'dateAdded', 'manufacturer', 'manufacturerNumber', 'asins', 'imageURLs', 'prices.sourceURLs', 'keys', 'sourceURLs']
df = df.drop(columns=removed)


# In[71]:


df.head()


# In[72]:


df['categories'].unique()


# In[73]:


df = df.drop('categories', axis = 1)


# In[74]:


df.head()


# In[75]:


# convert weight
def convert_to_kg(weight):
    if pd.isna(weight):
        return None
    # Convert to lower case and strip any whitespace
    weight = weight.lower().strip()
    
    # Check for different units and convert to kg
    if 'pounds' in weight or 'pound' in weight or 'lb' in weight:
        # Extract numeric value and convert to kg
        pounds = float(re.findall(r'\d*\.?\d+', weight)[0])
        return pounds * 0.453592
    elif 'ounces' in weight or 'ounce' in weight or 'oz' in weight:
        # Extract numeric value and convert to kg
        ounces = float(re.findall(r'\d*\.?\d+', weight)[0])
        return ounces * 0.0283495
    elif 'kg' in weight:
        # Extract numeric value (already in kg)
        return float(re.findall(r'\d*\.?\d+', weight)[0])
    elif 'g' in weight:
        # Extract numeric value and convert to kg
        grams = float(re.findall(r'\d*\.?\d+', weight)[0])
        return grams / 1000.0
    else:
        return None


# In[76]:


df['weight'] = df['weight'].apply(convert_to_kg)


# In[77]:


df['weight'].unique


# In[78]:


df = df.dropna()


# In[79]:


df.isnull().sum()


# In[80]:


df.head()


# In[44]:


#correlation matrix
import seaborn as sns
correlation_matrix = df.corr()
plt.figure(figsize = (15,15))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm')
plt.show()


# In[81]:


#printing correlation variables with the target variable
print(correlation_matrix['price'].sort_values(ascending = False))


# In[82]:


# Normalize the price
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])


# In[83]:


##EDA
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (10,5))
plt.scatter(df.index, df['price']) #a seaborn scatter plot to visualize price
plt.title('Scatter plot of Price')
plt.xlabel('index')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[84]:


##EDA
#analysing the relationship between price and weight

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'weight', y = 'price', data = df) #a seaborn scatter plot to visualize weight against data
plt.title('weight vs. Price')
plt.xlabel('weight')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[85]:


##EDA
#analysing the relationship between price and prices.condition_New

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'prices.condition_New', y = 'price', data = df) #a seaborn scatter plot to visualize weight against data
plt.title('prices.condition_New vs. Price')
plt.xlabel('prices.condition_New')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[86]:


##EDA
#analysing the relationship between price and prices.availability_Special Order

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'prices.availability_Special Order', y = 'price', data = df) #a seaborn scatter plot to visualize weight against data
plt.title('prices.availability_Special Order vs. Price')
plt.xlabel('prices.availability_Special Order')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[87]:


##EDA
#analysing the relationship between price and prices.availability_Yes

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'prices.availability_Yes', y = 'price', data = df) #a seaborn scatter plot to visualize weight against data
plt.title('prices.availability_Yes vs. Price')
plt.xlabel('prices.availability_Yes')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[88]:


##EDA
#analysing the relationship between price and primaryCategories_Electronics,Furniture

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'primaryCategories_Electronics,Furniture', y = 'price', data = df) #a seaborn scatter plot to visualize weight against data
plt.title('primaryCategories_Electronics,Furniture vs. Price')
plt.xlabel('primaryCategories_Electronics,Furniture')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[89]:


##EDA
#analysing the relationship between price and prices.isSale

plt.figure(figsize = (10,5))
sns.scatterplot(x= 'prices.isSale', y = 'price', data = df) #a seaborn scatter plot to visualize weight against data
plt.title('prices.isSale vs. Price')
plt.xlabel('prices.isSale')  # sets the x label of the plot to be 'weight'
plt.ylabel('Price($)') # sets the y label of the plot to be 'weight'
plt.show()


# In[90]:


#using robust scaler to remove outliers in the target variables
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
columns_to_scale = ['price', 'weight', 'prices.condition_New','prices.availability_Special Order', 'prices.availability_Yes', 'primaryCategories_Electronics,Furniture', 'prices.isSale']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
print(df)


# In[58]:


print(df[columns_to_scale])


# In[61]:


#using standard scaler to remove outliers in the target variables
#from sklearn.preprocessing import StandardScaler
#ss = StandardScaler()
#columns_= ['price', 'weight', 'prices.condition_New','prices.availability_Special Order', 'prices.availability_Yes', 'primaryCategories_Electronics,Furniture', 'prices.isSale']
#df[columns_] = ss.fit_transform(df[columns_])
#print(df)


# In[91]:


#correlation matrix
import seaborn as sns
correlation_matrix2 = df.corr()
plt.figure(figsize = (15,15))
sns.heatmap(correlation_matrix2, annot = True, cmap = 'coolwarm')
plt.show()


# In[94]:


print(correlation_matrix2['price'].sort_values(ascending = False))


# In[95]:


from sklearn.ensemble import RandomForestRegressor


# In[96]:


#create model
model = RandomForestRegressor()


# In[98]:


X = df[['weight', 'prices.condition_New', 'prices.availability_Special Order', 'prices.availability_Yes', 'primaryCategories_Electronics,Furniture', 'prices.isSale']]
X = X[:int(len(df)-1)]


# In[99]:


y = df['price']
y = y[:int(len(df)-1)]


# In[100]:


model.fit(X,y)


# In[101]:


#test the model
predictions = model.predict(X)


# In[102]:


print("The model's score is:  ", model.score(X,y))


# In[103]:


#making predictions
new_data = df[['weight', 'prices.condition_New', 'prices.availability_Special Order', 'prices.availability_Yes', 'primaryCategories_Electronics,Furniture', 'prices.isSale']].tail(1)


# In[104]:


prediction = model.predict(new_data)


# In[105]:


print("The model predicts the last price to be: ", prediction)
print("Actual value is: ", df[['price']].tail(1).values[0][0])


# In[ ]:




