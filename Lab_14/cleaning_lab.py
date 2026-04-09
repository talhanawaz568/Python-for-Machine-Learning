import pandas as pd
import numpy as np

# --- Task 1: Identify Missing or Duplicate Values ---
print("Step 1: Loading Messy Data...")
df = pd.read_csv('example_data.csv')

print("\nInitial Missing Values:")
print(df.isnull().sum())

print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# --- Task 2: Handling Missing and Duplicates ---
print("\nStep 2: Cleaning the Data...")

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values: 
# We'll fill NumericColumn with the average and old_name2 with 'Unknown'
df['NumericColumn'] = pd.to_numeric(df['NumericColumn'], errors='coerce') # Ensure numeric
mean_age = df['NumericColumn'].mean()

df = df.fillna({
    'NumericColumn': mean_age,
    'old_name2': 'Unknown'
})

# If any rows are still completely empty (like the one with just Chicago), we drop them
df = df.dropna(subset=['old_name1'])

# --- Task 3: Formatting Columns ---
print("Step 3: Formatting Columns...")

# 3.1 Strip Whitespace (Alice had extra spaces)
df['old_name1'] = df['old_name1'].str.strip()

# 3.2 Change Data Types (Ensure age is float)
df['NumericColumn'] = df['NumericColumn'].astype(float)

# 3.3 Rename Columns
df = df.rename(columns={
    'old_name1': 'Name', 
    'old_name2': 'City',
    'NumericColumn': 'Age'
})

# --- Final Output ---
print("\n" + "="*30)
print("CLEANED DATAFRAME")
print("="*30)
print(df)
print("\nFinal Missing Values Check:")
print(df.isnull().sum())


