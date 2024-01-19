
# coding: utf-8

# In[2]:


import pandas as pd

data = pd.read_csv('Banknote-authentication-dataset-.csv')

print(data)


# In[4]:


v1 = data['V1']
v2 = data['V2']

print('max V1: ', v1.max())
print('min V2: ', v2.min())
print("There are {:} rows ".format(data.shape[0]) + "and {} columns in our data".format(data.shape[1]))


# In[6]:


data.info


# In[8]:


data.describe()


# In[10]:


data.describe().sum()


# In[15]:


import matplotlib.pyplot as plt

plt.xlabel('v1')            #etiqueta el grafico
plt.ylabel('v2')

plt.scatter(v1, v2)


# In[19]:


from sklearn.cluster import KMeans #importa K-means
import numpy as np

prueba = np.column_stack((v1, v2))#crea dos columnas
km_res = KMeans(n_clusters=3).fit(prueba)

clusters = km_res.cluster_centers_

plt.scatter(v1, v2)
plt.scatter(clusters[:,0], clusters[:,1], s=1000)


# In[23]:




