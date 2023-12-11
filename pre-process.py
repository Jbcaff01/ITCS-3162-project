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
plt.ylabel(y_column)

# %%
genre_sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=genre_sales.values, y=genre_sales.index)
plt.title('Total Global Sales by Genre')
plt.xlabel('Global Sales (in millions)')
plt.ylabel('Genre')
plt.show()

# %%

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

# %%
label_encoder = LabelEncoder()
df['Rating'] = label_encoder.fit_transform(df['Rating'])

# %%
X = df[['Year_of_Release', 'User_Score', 'Rating']]
y = df['Global_Sales']

# %%
custom_rating_order = ['E', 'E10+', 'T', 'M', 'AO']
df['Rating'] = pd.Categorical(df['Rating'], categories=custom_rating_order, ordered=True)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
comparison_df = pd.DataFrame({'Rating': label_encoder.inverse_transform(X_test['Rating']),
                              'Actual_Sales': y_test,
                              'Predicted_Sales': y_pred})

# %%
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Actual_Sales', y='Predicted_Sales', hue='Rating', data=comparison_df)
plt.title('Actual vs Predicted Global Sales by Rating')
plt.xlabel('Actual Global Sales')
plt.ylabel('Predicted Global Sales')
plt.legend(title='Rating')
plt.show()

# %%
label_encoder_platform = LabelEncoder()
label_encoder_genre = LabelEncoder()
label_encoder_rating = LabelEncoder()

# %%
df['Platform'] = label_encoder_platform.fit_transform(df['Platform'])
df['Genre'] = label_encoder_genre.fit_transform(df['Genre'])
df['Rating'] = label_encoder_rating.fit_transform(df['Rating'])

# %%
X = df[['Year_of_Release', 'User_Score', 'Platform', 'Genre', 'Rating']]
y = df['Global_Sales']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# %%
y_pred = rf_model.predict(X_test)

comparison_df = pd.DataFrame({'Platform': label_encoder_platform.inverse_transform(X_test['Platform']),
                              'Actual_Sales': y_test,
                              'Predicted_Sales': y_pred})

platform_avg_sales = comparison_df.groupby('Platform')['Predicted_Sales'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=platform_avg_sales.index, y=platform_avg_sales.values)
plt.title('Average Predicted Global Sales by Platform')
plt.xlabel('Platform')
plt.ylabel('Average Predicted Global Sales')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
X = df[['Year_of_Release', 'User_Score', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Platform', 'Genre', 'Publisher', 'Rating']]
y = df['Global_Sales']
X = pd.get_dummies(X, columns=['Platform', 'Genre', 'Publisher', 'Rating'])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('K Nearest Neighbor Model')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# %%
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Random Forest Model')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


