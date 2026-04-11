import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for headless Ubuntu environments to save plots as files
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for splitting data and modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# --- Task 1: Load and Inspect Dataset ---
print("Step 1: Loading California Housing Data...")
housing = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

# Inspecting the data
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

# --- Task 2: Split Features and Target ---
print("\nStep 2: Splitting data into Training (80%) and Testing (20%)...")
X = df.drop('PRICE', axis=1)
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Task 3: Train and Evaluate Model ---
print("Step 3: Training the Linear Regression model...")
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

# Make Predictions
y_pred = lin_reg_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 40)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score:    {r2:.4f}")
print("-" * 40)

# --- Visualization ---
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.xlabel('Actual Prices ($100k units)')
plt.ylabel('Predicted Prices ($100k units)')
plt.title('Actual vs Predicted House Prices')

# Draw the 'Perfect Prediction' line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)

plt.grid(True)
plt.savefig('linear_regression_results.png')
print("\n✓ Results visualization saved as 'linear_regression_results.png'")
