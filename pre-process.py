#!/usr/bin/env python
# coding: utf-8

# # Video Game Sales

# In[13]:


import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt


# ## Data Introduction

# In[14]:


df = pd.read_csv('Video_Game_Sales_as_of_Jan_2017.csv')


# In[15]:


df.head()


# In[25]:


df.shape


# In[17]:


df.isna().sum()


# ## Pre_Processing

# ### Dropping Critic_Score, User_Count and Critic_Count as it is not relevant to our problem. We will drop the NaN values in User_Score. 

# In[19]:


df = df.drop(columns=['Critic_Score', 'Critic_Count', 'User_Count'])


# In[21]:


df = df.dropna()


# ### Changing Year_of_Release data type from float to int

# In[28]:


df = df.astype({"Year_of_Release": int})



# In[29]


sns.pairplot(data=df,y_vars='Global_Sales',kind='hist')

# In[30]

sns.pairplot(data=df,y_vars='Global_Sales',x_vars='Publishers',kind='bar')


