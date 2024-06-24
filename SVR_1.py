#!/usr/bin/env python
# coding: utf-8

# In[24]:


get_ipython().system('pip list')


# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


import pandas as pd
df = pd.read_csv('C:\\Users\\JOAN\\group1_task_zummit\\electronics_products_pricing.csv') #reading in the data


# In[3]:


df.head()


# In[4]:


# removing columns
removed = ['id', 'ean', 'categories', 'prices.shipping', 'upc', 'name', 'dateUpdated', 'brand', 'prices.merchant', 'prices.dateSeen', 'dateAdded', 'manufacturer', 'manufacturerNumber', 'asins', 'imageURLs', 'prices.sourceURLs', 'keys', 'sourceURLs']
df = df.drop(columns=removed, errors = 'ignore')


# In[5]:


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


# In[6]:


df['weight'] = df['weight'].apply(convert_to_kg)


# In[7]:


# Convert categorical data to numerical data
df['prices.isSale'] = df['prices.isSale'].astype(int)
columns= ['prices.availability', 'primaryCategories', 'prices.condition', 'prices.currency']
df = pd.get_dummies(df, columns=columns)
df.head()


# In[8]:


df.isnull().sum()  
df = df.dropna()


# In[9]:


df.isnull().sum()


# In[10]:


remove =['prices.condition_5/16" Ring Terminal, 3 ft. 8 GA Black Ground Cable, 6 ft. Split Loom Tubing, Depth: 6.5" (165mm) (top) 11.2" (285mm) (bottom), Item Weight: 18.5 lbs., Frequency Response 25Hz - 500Hz, Line Output, Max Power: 1100 Watt x 1 Channel @ 2 Ohm, 30 ft. Speaker Wire, Boss Illuminated Logo, Height: 12.8" (325mm), (3) Rubber Grommets, Item Weight: 2 lbs., Size 10", 20 ft. 8 GA Red Power Cable, Ported enclosure for greater output and deeper bass, 2 Ohm Stable, Class A/B, Voice Coil Size 2", Black rubber surround, Nominal Impedance 4 ohm, Rugged fiberboard construction with thick carpet covering, Warranty: 1 Year Manufacturer Warranty, MOSFET Power, Weight: 6 lbs, Width: 17.7" (450mm), Condition: Brand New!, Low Pass Crossover, List item, RMS Power: 250 Watt x 1 Channel @ 4 Ohm, Remote Bass Control Included!, 1/4" Ring Terminal, 16 ft. 18 GA Blue Turn-On Wire, Peak Power: 500 Watts, Competition High Quality Fuse Holder, Condition: BRAND NEW!, Product Condition: Brand New, RMS Power: 175 Watts, Aluminum woofer cone, THD: 0.01%, 1 Year Manufacturer Warranty, Dimensions: 10-7/16" (W) x 2-1/4" (H) x 9-1/8" (L), #10 Ring Terminal, 20 ft. High Performance Black RCA, SPL (db @ 1w/1m) 88dB, New Dual SBX101 10" 500 Watts Car Audio Subwoofer Sub + Ported Sub Enclosure, (20) 4" Wire Ties', 'prices.condition_New Kicker BT2 41IK5BT2V2 Wireless Bluetooth USB Audio System Black + Remote, Power Supply (volts, ampere): 24, 2.9, Square Reflex Subwoofer (in., mm): 6 x 6", Stereo Amp Power with DSP (watts): 50, App for customizing - KickStart, Remote Control Included, Height x Width x Depth (in, mm): 8.87" x 19" x 9.14", Frequency Response (Hz): 24-20k, +-3dB, Woofer (in., cm): 5", 1 Year Manufacturer Warranty, Item Weight: 13.85 lbs., USB Port, Compatible with: Bluetooth-enabled devices, Aux-in, Speaker Design: 2-Way Full-Range, Bluetooth wireless streaming, Condition: Brand New!, Tweeter (in., cm): 3/4"']
df = df.drop(columns = remove, errors = 'ignore')


# In[11]:


df.columns


# In[12]:


rename_dict = {'prices.availability_7 available': 'prices.available', 'prices.availability_In Stock': 'prices.availablity_In_Stock', 'prices.availability_More on the Way': 'prices.availability_More_on_the_Way', 'prices.availability_Out Of Stock': 'prices.availability_Out_of_Stock', 'prices.availability_Special Order': 'prices.availability_Special_Order', 'primaryCategories_ Apple CarPlay': 'primaryCategories_Apple_CarPlay', 'primaryCategories_ Intel Celeron': 'primaryCategories_Intel_Celeron', 'primaryCategories_ Siri Eyes Free': 'primaryCategories_SiriEyesFree', 'prices.condition_New other (see details)': 'prices.condition_New_other', 'prices.condition_Seller refurbished': 'prices.condition_Seller_refurbished', 'prices.condition_pre-owned': 'prices.condition_pre_owned'}
df.rename(columns = rename_dict, inplace = True)

print(df)


# In[13]:


# Normalize the price
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])


# In[14]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
X = df.drop(columns = ['price'], axis = 1)
X = X[:int(len(df)-1)]

y = df['price']
y = y[:int(len(df)-1)]
#defining the logarithmic normalization function 
log_transformer = FunctionTransformer(np.log1p, validate = True)

#applying the logarithmic normalization to the selected features
X_log_trasnformed = log_transformer.fit_transform(X)


# In[17]:


X


# In[18]:


#using robust scaler to remove outliers in the target variables
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled= scaler.fit_transform(X_log_trasnformed)


# In[19]:


print(len(X_scaled), len(y))


# In[27]:


X_scaled


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)


# In[29]:


svr_model = SVR(kernel = 'rbf')
svr_model.fit(X_train, y_train)


# In[31]:


y_pred = svr_model.predict(X_test)


# In[32]:



mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


# In[33]:


print("Mean Squared Error: ", mse)
print("R-squared: ", r2)


# In[34]:


print("The model's score is:  ", svr_model.score(X_train,y_train))


# In[35]:


new_data = df.drop(columns = ['price'], axis =1).tail(1)


# In[36]:


prediction = svr_model.predict(new_data)


# In[37]:


print("The model predicts the last price to be: ", prediction)
print("Actual value is: ", df[['price']].tail(1).values[0][0])


# In[38]:


X_scaled_ = np.array([['prices.isSale', 'weight', 'prices.available', 'prices.availability_FALSE', 'prices.availability_In_Stock', 'prices.availability_More_on_the_Way', 'prices.availability_No', 'prices.availability_Out_Of_Stock', 'prices.availability_Retired', 'prices.availability_Special_Order', 'prices.availability_TRUE', 'prices.availability_Yes', 'prices.availability_sold', 'prices.availability_undefined', 'prices.availability_yes', 'primaryCategories_Apple_CarPlay', 'primaryCategories_Intel_Celeron', 'primaryCategories_SiriEyesFree', 'primaryCategories_Electronics', 'primaryCategories_Electronics,Furniture', 'prices.condition_New', 'prices.condition_New_other', 'prices.condition_Refurbished ', 'prices.condition_Seller_refurbished', 'prices.condition_Used', 'prices.condition_new', 'prices.condition_pre_owned', 'prices.condition_refurbished', 'prices.currency_CAD', 'prices.currency_USD']])


# In[39]:


import pickle


# In[41]:


file_path = 'C:\\Users\\JOAN\\OneDrive\\Desktop\\ML_PROJECT_1\\svr_model.pkl'

with open(file_path, 'wb') as file:
    pickle.dump(svr_model, file)


# In[ ]:




