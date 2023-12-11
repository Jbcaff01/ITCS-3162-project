#!/usr/bin/env python
# coding: utf-8

# # Video Game Sales

# In[6]:


import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt


# ## Data Introduction

# In[14]:


df = pd.read_csv('Video_Game_Sales_as_of_Jan_2017.csv')


# In[20]:


df.head()
df.dtypes


# In[26]:


df.shape


# In[32]:


df.isna().sum()


# ## Pre_Processing

# ### Dropping Critic_Score, User_Count and Critic_Count as it is not relevant to our problem. We will drop the NaN values in User_Score. 

# In[42]:


df = df.drop(columns=['Critic_Score', 'Critic_Count', 'User_Count'])


# In[48]:


df = df.dropna()


# ### Changing Year_of_Release data type from float to int

# In[56]:


df = df.astype({"Year_of_Release": int})



# In[63]

sns.pairplot(data=df,y_vars='Global_Sales',kind='hist')

# In[67]

selected_column ='Year_of_Release'
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
plt.hist(df[selected_column], bins=10, color='skyblue', edgecolor='black')
plt.title(f'Histogram for {selected_column}')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()



#In[83]

# In[67]

selected_column ='Platform'

sns.set(style="whitegrid")

plt.figure(figsize=(10, 10))
plt.hist(df[selected_column], bins=15, color='skyblue', edgecolor='black')
plt.title(f'Histogram for {selected_column}')
plt.xlabel('Pllatform')
plt.ylabel('Frequency')
plt.show()

# In[98]

selected_column ='Genre'
sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
plt.hist(df[selected_column], bins=10, color='skyblue', edgecolor='black')
plt.title(f'Histogram for {selected_column}')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.show()


# In[108]

x_column = 'Rating'
y_column = 'Genre'

# Plot a scatter plot
plt.scatter(df[x_column], df[y_column])

# Customize the plot if needed
plt.title(f'Scatter Plot: {x_column} vs {y_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)()

# %%
