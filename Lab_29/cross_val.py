import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Task 1.3: Loading the Dataset ---
print("Step 1: Loading Iris Dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# --- Task 1.4: Implementing k-Fold Cross-Validation ---
print("\nStep 2: Starting 5-Fold Cross-Validation...")

# Define the number of folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Increase max_iter to 200 to ensure LogisticRegression converges
model = LogisticRegression(max_iter=200)

accuracies = []
fold_number = 1

# The kf.split() function gives us the indices for each rotation
for train_index, test_index in kf.split(X):
    # Partitioning the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model on k-1 folds
    model.fit(X_train, y_train)
    
    # Test on the remaining fold
    predictions = model.predict(X_test)
    
    # Calculate accuracy for this specific rotation
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)
    
    print(f"Fold {fold_number} Accuracy: {accuracy:.4f}")
    fold_number += 1

# --- Task 2: Evaluate Performance Across Folds ---
print("\n" + "="*35)
print("FINAL EVALUATION")
print("="*35)

average_accuracy = np.mean(accuracies)
standard_deviation = np.std(accuracies)

print(f"Accuracies for each fold: {['{:.2f}'.format(a) for a in accuracies]}")
print(f"Average accuracy across {k} folds: {average_accuracy:.4f}")
print(f"Variance (Std Dev) across folds: {standard_deviation:.4f}")

# --- Task 3: Practical Shortcut (Pro Tip) ---
# In real projects, we usually use the cross_val_score function 
# instead of a manual loop. It does all the work above in one line:
# scores = cross_val_score(model, X, y, cv=5)
# print(f"Shortcut Result: {scores.mean():.4f}")
