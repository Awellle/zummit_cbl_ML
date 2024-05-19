#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[2]:


import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv('c:\\users\\joan\\Downloads\\zoo animals dataset\\zoo2.csv')


# In[5]:


df2 = pd.read_csv('c:\\users\\joan\\Downloads\\zoo animals dataset\\zoo3.csv')


# for the k-means clustering, i will be using only one dataset. i.e 'df'... as i have discovered that merging the two datasets doesn't work well with the algorithm.

# In[6]:


#using the fewature 'hair' for clustering
#normalizing the data. I'll be selecting the column 'hair' as the feature to be used for clustering
from sklearn.preprocessing import StandardScaler
X1 = df.values[:,1:]  # df.values converts the dataframe into a NumPy array, '[:, 1]' selects all rows(':') of the 2nd column in the DF. this line assumes i'm interested in standardizing this particular feature 'hair' across all entries. 
X1= np.nan_to_num(X1)
Clus_Set = StandardScaler().fit_transform(X1)
Clus_Set


# In[7]:


cluster_num = 3
k_means = KMeans(init = "k-means++", n_clusters = cluster_num, n_init = 12)
k_means.fit(X1)
labels = k_means.labels_
print(labels)


# In[8]:


df["Cluster_labels"] = labels
df.head(5)


# In[9]:


df.groupby('Cluster_labels').mean()


# In[20]:


#looking at thedistribution of animals based on their hair and teeth
area = np.pi * ( X1[:, 1])**2  
plt.scatter(X1[:, 0], X1[:, 6], s=area, c=labels.astype(), alpha=0.5)
plt.xlabel('hair', fontsize=18)
plt.ylabel('toothed', fontsize=16)

plt.show()


# In[6]:


concat_df = pd.concat([df,df2], ignore_index= True)


# In[7]:


concat_df.drop_duplicates()


# In[8]:


#shuffling the data after concatenation to ensure the model does not learn any unintended patterns from the order in which it was merged

shuffled_df = concat_df.sample(frac = 1, random_state = 42).reset_index(drop = True)


# In[9]:


shuffled_df


# In[10]:


#dropping the 'class_type' column
new_df = shuffled_df.drop(columns = ['class_type'])
new_df.head()


# In[11]:


new_df.describe()


# In[12]:


new_df.info()  #checking for null values. There are no null values


# In[26]:


#normalizing the data. I'll be selecting the colum 'legs' as the feature to be used for clustering
#from sklearn.preprocessing import StandardScaler
#X = df.values[:,13:]  # df.values converts the dataframe into a NumPy array, '[:, 13]' selects all rows(':') of the 14th column in the DF. this line assumes i'm interested in standardizing this particular feature 'legs' across all entries. 
#X= np.nan_to_num(X)
#Cluster_Set = StandardScaler().fit_transform(X)
#Cluster_Set


# In[21]:


modeling 
#cluster_number = 3
#kmeans = KMeans(init = "k-means++", n_clusters = cluster_number, n_init = 12)
#kmeans.fit(X)
#labels = kmeans.labels_
#print(labels) i noticed that when i used this feature, i only got 43 values, instead of 113


# In[28]:


#using another feature 'hair' for clustering
#normalizing the data. I'll be selecting the column 'hair' as the feature to be used for clustering
from sklearn.preprocessing import StandardScaler
X1 = df.values[:,1:]  # df.values converts the dataframe into a NumPy array, '[:, 1]' selects all rows(':') of the 2nd column in the DF. this line assumes i'm interested in standardizing this particular feature 'hair' across all entries. 
X1= np.nan_to_num(X1)
Clus_Set = StandardScaler().fit_transform(X1)
Clus_Set


# In[31]:


#modeling 
cluster_number = 3
kmeans = KMeans(init = "k-means++", n_clusters = cluster_number, n_init = 12)
kmeans.fit(X1)
labels1 = kmeans.labels_
print(labels1) 


# In[33]:


from sklearn.preprocessing import StandardScaler
X2 = df.values[:,10:]  # df.values converts the dataframe into a NumPy array, '[:, 10]' selects all rows(':') of the 11th column in the DF. this line assumes i'm interested in standardizing this particular feature 'breathes' across all entries. 
X2= np.nan_to_num(X2)
Clus_Set = StandardScaler().fit_transform(X2)
Clus_Set


# In[34]:


cluster_number = 3
kmeans = KMeans(init = "k-means++", n_clusters = cluster_number, n_init = 12)
kmeans.fit(X2)
labels2 = kmeans.labels_
print(labels2) 


# In[18]:


#creating a new column which will contain the labels. in other words, assigning the labels to each row in the dataset

new_df["cluster_labels"] = labels


# observation: the length of the labels does not match the length of the index, when i use one of the dataframes and when i merge them.

# ### Hierarchical clustering(agglomerative)

# In[32]:


new_df.head()


# In[36]:


#when i checked the info (i.e new_df.info), there were no missing values. however, ill still try to clear missing values, to be thorough

print ("Shape of dataset before cleaning: ", new_df.size)
new_df[['hair', 'feathers', 'eggs', 'milk',
       'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes',
       'venomous', 'fins']] = new_df[['hair', 'feathers', 'eggs', 'milk',
       'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes',
       'venomous', 'fins']].apply(pd.to_numeric, errors='coerce')  # 'errors = corece' instructs pandas to handle errors by converting problematic values to Nan/Not a Number
new_df = new_df.dropna()
new_df = new_df.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", new_df.size)
new_df.head(5)


# In[47]:


# the result above confirms that there were no null values initially


# In[38]:


#selecting the feature set
features = new_df[['feathers', 'eggs', 'predator', 'backbone', 'breathes', 'legs', 'fins']]


# In[40]:


#normalizing 
from sklearn.preprocessing import MinMaxScaler
X4 = features.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_matrix = min_max_scaler.fit_transform(X4)
feature_matrix [0:5]


# In[41]:


#clustering using scikit-learn
from sklearn.metrics.pairwise import euclidean_distances
distance_matrix = euclidean_distances(feature_matrix,feature_matrix) 
print(distance_matrix)


# In[46]:


import pylab
from scipy.cluster import hierarchy 
import scipy.cluster.hierarchy
Z = hierarchy.linkage(distance_matrix, 'complete')


# In[48]:


fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (new_df['animal_name'][id], new_df['fins'][id], new_df['domestic'][id])
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =10, orientation = 'right')


# In[50]:


#clustering the dataset using agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(distance_matrix)

agglom.labels_


# In[51]:


#creating a column to hold the labels and essentially, assigning the labels to each data point
new_df["cluster_labels"] = agglom.labels_
new_df.head()


# In[71]:


import matplotlib.cm as cm #for managing color maps in matplotlibs
n_clusters = max(agglom.labels_)+1 #calculates the max. number of clusters identified by agglo.labels_and adds one bc cluster labels typically start at 0
colors = cm.rainbow(np.linspace(0, 1, n_clusters)) #generates a list of colrs by creating an evenly spaced array of values from 0 to 1, corresponding to the number of clusters, and then maps these values to colors in a rainbow color map.
clust_labels = list(range(0, n_clusters)) # creates a list of integers representing each cluster, ranging from 0 to the max. number of clusters. 

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14)) # sets up a new figure with a size of 16*4 inches to plot the data

# loop to plot each cluster's data 
for color, label in zip(colors, clust_labels): # outer for loop, iterates over each cluster, using a color and a label. 
    subset = new_df[new_df.cluster_labels == label] # filters the main dataframe 'new_df' to include only the rows where the cluster label matches the current label
    for i in subset.index: # inner for loop, iterates over each item in the filtered subset
            plt.text(subset.feathers[i], subset.eggs[i], subset.backbone[i], rotation = 25) 
    plt.scatter(subset.feathers, subset.eggs, s= subset.legs*300, c=color, label='cluster'+str(label),alpha=0.5) # plots each data point on the scaatter plot. The x-coordinate is 'feathers', the y-coordinate is 'eggs', and the size of the dot is proportional to the column 'legs', the color is consistent and the transparency 'alpha' is set to 0.5 for better visibility of overlapping points
plt.legend() # adds a legend to the plot
plt.title('Clusters') # sets the tile of the plot
plt.xlabel('feathers')# sets a label for the x-axis
plt.ylabel('eggs') #sets a label for the y-axis


# ### DBSCAN

# In[21]:


df.head()


# In[22]:


import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets import make_blobs #makea-blobs generates data points and their corresponding labels based on the parameters specified
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[30]:


#checking the data(for null values)

df.info()


# In[31]:


df2.head()


# In[32]:


df2.info()


# In[33]:


get_ipython().system('pip install basemap')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = (12,10)

llon=-140  #lower longitude, is the minimum longitude value, indicating the western boundary of the map
ulon=-50 #upper longitude is the max. longitude value, indicating the eastern boundary of the map
llat=40 #lower latitude is the min latutude value, indicating the southern boundary of the map
ulat=65 #upper latitude, is the max. latitude value, indicating the northern boundary of the map

df2 = df2[(df['hair'] > llon) & (df2['hair'] < ulon) & (df2['feathers'] > llat) & (df2['feathers'] < ulat)]

my_map = Basemap(projection='merc',
            resolution = 'l', #sets the resolution of the map to low, area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()

# To collect data based on         

xs,ys = my_map(np.asarray(df.hair), np.asarray(df.feathers))
df2['xm']= xs.tolist()
df2['ym'] =ys.tolist()

#Visualization1
for index,row in df.iterrows():
    my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
#plt.text(x,y,stn)
plt.show()


# In[34]:


# clustering the animals based on whether they have feathers or hair i.e mammals or birds
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = df2[['xm','ym']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
df["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 


# A sample of clusters
df2[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)


# In[35]:


df2['hair'].shape


# observation: i noticed that it keeps saying it found array with shape in thr form of (0,)... i have tried using both dataframes and still got this error. 
