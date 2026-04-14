import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for Ubuntu/Headless environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# --- Task 1: Load and Explore ---
print("Task 1: Loading Dataset...")
df = pd.read_csv('sample_data.csv')
print(df.head())

# --- Task 2: Create New Features ---
print("\nTask 2: Performing Feature Transformations...")

# 2.1 Transformations (Log and Square)
# Log transforms are great for skewed data (making it look like a normal distribution)
df['area_log'] = np.log1p(df['area'])
df['age_square'] = df['age'] ** 2

# 2.2 Interaction Features
# Combining features often reveals new logic. 
# Example: Area per room gives a sense of "Room Size"
df['area_per_room'] = df['area'] / df['rooms']

print(f"New Features Created. Total columns: {len(df.columns)}")

# --- Task 3: Assess Relevance ---
print("\nTask 3: Assessing Feature Relevance...")

# 3.1 Correlation Matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig('correlation_heatmap.png')
print("✓ Saved: correlation_heatmap.png")

# 3.2 Feature Importance with Random Forest
X = df.drop('target', axis=1)
y = df['target']

model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X, y)

# Create a DataFrame for importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n--- Feature Importance Results ---")
print(importance_df)

# Final Plot for Feature Importance
plt.figure(figsize=(8, 6))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title("Random Forest Feature Importance")
plt.savefig('feature_importance.png')
print("✓ Saved: feature_importance.png")
