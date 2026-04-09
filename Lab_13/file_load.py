import pandas as pd
import os

# --- Task 1: Load a CSV File ---
file_path = 'data.csv'

# Check if the file exists before trying to load it
if os.path.exists(file_path):
    print(f"Step 1: Loading {file_path}...")
    df = pd.read_csv(file_path)
else:
    print(f"Error: {file_path} not found. Please create it first!")
    exit()

# --- Task 2: Display the First Few Rows ---
print("\nStep 2: Previewing the data (head):")
# We use head() to see if our columns aligned correctly
print(df.head())

# --- Task 3: Check for Missing Values ---
print("\nStep 3: Checking for Missing (NaN) Values:")
# isnull() creates a table of True/False, sum() counts the Trues
missing_data = df.isnull().sum()

print("Missing values per column:")
print(missing_data)

# --- Analysis ---
print("\n" + "="*30)
print("LAB ANALYSIS")
print("="*30)
total_cells = df.size
missing_cells = missing_data.sum()
print(f"Total data points: {total_cells}")
print(f"Missing data points: {missing_cells}")
print("Note: 'NaN' stands for 'Not a Number' in Pandas.")
