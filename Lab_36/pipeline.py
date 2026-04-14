import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# --- Task 1: Build a Pipeline with Preprocessing Steps ---
print("Task 1: Initializing Preprocessing...")

# Step 2: Load Data
data = load_iris()
X, y = data.data, data.target

# Step 3 & 4: Define and Create Preprocessing Pipeline
# SimpleImputer fills missing values (if any)
# StandardScaler ensures all features have a mean of 0 and variance of 1
preprocessing_steps = [
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
]
preprocessing_pipeline = Pipeline(preprocessing_steps)

# --- Task 2: Integrate Model Training ---
print("Task 2: Integrating Logistic Regression Model...")

# Step 6: Create the full Model Pipeline
# Note: We can nest the preprocessing_pipeline inside the final model_pipeline
model_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('classifier', LogisticRegression())
])

# Step 7: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Task 3: Evaluate the Complete Pipeline ---
print("\nTask 3: Evaluating Performance...")

# Step 8: Evaluate with Cross-Validation
# This tests the entire pipeline (scaling + model) 5 different times
scores = cross_val_score(model_pipeline, X_train, y_train, cv=5)

print("-" * 30)
print(f"Cross-validation scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.2f}")
print("-" * 30)

# Step 9: Final Train and Test
model_pipeline.fit(X_train, y_train)
test_score = model_pipeline.score(X_test, y_test)
print(f"Final Test Set Accuracy: {test_score:.2f}")

print("\n✓ Lab 36 Complete: Pipeline is ready for deployment.")
