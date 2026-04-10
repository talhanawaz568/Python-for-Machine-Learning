import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for headless Ubuntu environment
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- Task 2: Scale a sample dataset ---
print("Task 2: Preparing and Scaling Data...")

# Sample data: Feature1 could be Height (cm), Feature2 could be weight (g)
data = {'Feature1': [140, 150, 155, 160, 165],
        'Feature2': [2000, 2100, 2150, 2200, 2250]}

df = pd.DataFrame(data)
print("\n--- Original Data ---")
print(df)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
# 'Fit' calculates the mean and std dev; 'Transform' applies the formula
scaled_data = scaler.fit_transform(df)

# Convert the scaled data back into a DataFrame for easier reading
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print("\n--- Scaled Data (Mean ~0, Std ~1) ---")
print(scaled_df)

# --- Task 3: Compare scaled vs. unscaled data ---
print("\nTask 3: Analyzing Statistics...")

print("\nOriginal Data Statistics (Mean is large):")
print(df.describe().loc[['mean', 'std']])

print("\nScaled Data Statistics (Mean is zero):")
print(scaled_df.describe().loc[['mean', 'std']])

# --- Visualization ---
print("\nGenerating comparison charts...")

# Original Data Plot
plt.figure(figsize=(10, 5))
df.plot(kind='bar', title='Original Data (Different Scales)')
plt.ylabel('Value')
plt.savefig('original_data_bars.png')

# Scaled Data Plot
plt.figure(figsize=(10, 5))
scaled_df.plot(kind='bar', title='Scaled Data (Standardized)')
plt.ylabel('Z-Score')
plt.savefig('scaled_data_bars.png')

print("✓ Comparison charts saved as 'original_data_bars.png' and 'scaled_data_bars.png'")
