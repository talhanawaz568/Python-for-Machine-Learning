import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# --- Task 1: Use train_test_split ---
print("Task 1: Loading and Splitting Dataset...")
data = load_iris()
X = data.data
y = data.target

# Standard 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1.4 Verify the dimensions
print("-" * 30)
print(f"Total samples in original data: {len(X)}")
print(f"Training data shape (Features): {X_train.shape}")
print(f"Testing data shape (Features):  {X_test.shape}")
print("-" * 30)

# --- Task 3: Validate Split Ratios & Stratification ---
print("\nTask 3: Splitting with Stratification...")

# Stratify ensures the training and testing sets have the SAME
# percentage of each class (flower type) as the original data.
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Checking class distribution
print("Original class distribution: ", np.bincount(y))
print("Test set class distribution: ", np.bincount(y_test_s))
print("\n✓ Stratification complete: Distribution is balanced.")
