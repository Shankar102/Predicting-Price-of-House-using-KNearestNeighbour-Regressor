#!/usr/bin/env python
# coding: utf-8

#  https://github.com/edyoda/data-science-complete-tutorial/blob/master/Data/house_rental_data.csv.txt
# 
# 
# 
# Use pandas to get some insights into the data (10 marks)
# Show some interesting visualization of the data (10 marks)
# Manage data for training & testing (20)
# Finding a better value of k (10)
# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.datasets import make_blobs
import seaborn as sns


# In[2]:


df=pd.read_csv('data/House_Rental_Dataset.csv')
df


# In[3]:


df.isnull().sum()


# In[4]:


df.describe()


# In[5]:


i=df[df['Price']>125000] 
i


# In[6]:


TotalFloors=df.loc[df['TotalFloor']==6,'Price']
TotalFloors


# In[7]:



plt.figure(figsize=(10,8))
df['TotalFloor'].value_counts(normalize=True).plot.bar()
plt.xlabel('TotalFloors')
plt.ylabel('value counts')


# In[8]:


fig,ax=plt.subplots(figsize = (12,8) , layout='constrained')
ax.scatter('TotalFloor','Price',c='green',data=df)
plt.xlabel(' No. of TotalFloor')
plt.ylabel('Range of House Price')


# In[42]:


df=pd.read_csv('data/House_Rental_Dataset.csv')
df


# In[64]:


df=df.drop('Floor',axis=0)
df


# In[73]:


X=df.iloc[:, :-1].values
y=df.iloc[:, 4].values


# In[74]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


# In[79]:


from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy=[]

neighbors=range(1,11)
for number_of_neighbors in neighbors:
    KNN=KNeighborsClassifier(n_neighbors=number_of_neighbors)
    KNN.fit(X_train,y_train)
    training_accuracy.append(KNN.score(X_train,y_train))
    test_accuracy.append(KNN.score(X_test,y_test))


# In[81]:


plt.plot(neighbors,training_accuracy,label="training_accuracy")
plt.plot(neighbors,test_accuracy,label="test_accuracy")
print("K=2")


# In[ ]:




