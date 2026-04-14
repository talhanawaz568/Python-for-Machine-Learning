import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# --- Task 1: Train an SVM Model ---
print("Task 1: Loading and Preparing Dataset...")

# Step 1.2: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 1.3: Standardize Features 
# CRITICAL: SVMs calculate distances between points. 
# If features have different scales, the model will be biased.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 1.4: Train the SVM Model (RBF Kernel)
print("Training RBF Kernel Model...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# --- Task 2: Evaluate Performance ---
print("\nTask 2: Performance Evaluation (RBF)")
y_pred = svm_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Task 3: Experiment with Different Kernels ---
print("\nTask 3: Experimenting with Different Kernels...")

kernels = ['linear', 'poly', 'sigmoid']

for k in kernels:
    print(f"\n--- Performance: {k.upper()} KERNEL ---")
    if k == 'poly':
        model = SVC(kernel=k, degree=3, C=1.0) # Degree 3 for polynomial
    else:
        model = SVC(kernel=k, C=1.0)
        
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # We'll just show the overall accuracy for comparison
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy with {k} kernel: {accuracy:.4f}")
    
    if k == 'linear':
        print(classification_report(y_test, predictions))

print("\n✓ Lab 42 Complete: Check the outputs above to see which kernel performed best.")
