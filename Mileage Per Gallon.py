# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load the Dataset
data_path = 'path_to_auto_mpg_dataset.csv'  # Update the file path
df = pd.read_csv(data_path)

# 3. Initial Data Exploration
print(df.head())
print(df.describe())

# 4. Data Cleaning
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
df = pd.get_dummies(df, columns=['origin'], drop_first=True)
df.drop(columns=['car name'], inplace=True)

# 5. Data Visualization
sns.pairplot(df, x_vars=['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year'], y_vars='mpg', height=5)
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# 6. Feature Selection
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin_2', 'origin_3']
X = df[features]
y = df['mpg']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Training the Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# 9. Prediction using Test Data
y_pred = rf_model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
