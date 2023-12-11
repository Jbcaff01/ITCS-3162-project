# %%
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt

# %%
df = pd.read_csv('Video_Game_Sales_as_of_Jan_2017.csv')

# %%
df.head()

# %%
df.shape

# %%
df.isna().sum()

# %%
df = df.drop(columns=['Critic_Score', 'Critic_Count', 'User_Count'])

# %%
df = df.dropna()

# %%
df = df.astype({"Year_of_Release": int})

# %%
df.head()

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


