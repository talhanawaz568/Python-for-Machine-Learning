import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# --- Task 2: Data Preparation ---
print("Task 2: Preparing Dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Split the data (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Task 2.3: Implement Bagging ---
print("\nTask 2.3: Training Bagging Ensemble (10 Trees)...")
# We use a Decision Tree as the base model
base_model = DecisionTreeClassifier(random_state=42)

# n_estimators=10 means we are training 10 different trees
bagging_clf = BaggingClassifier(estimator=base_model, n_estimators=10, random_state=42)
bagging_clf.fit(X_train, y_train)

# Evaluate Bagging
y_pred_bag = bagging_clf.predict(X_test)
bagging_accuracy = accuracy_score(y_test, y_pred_bag)

# --- Task 3: Compare with Single Estimator ---
print("Task 3: Training Single Decision Tree for Comparison...")
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

# Evaluate Single Tree
y_pred_single = single_tree.predict(X_test)
single_accuracy = accuracy_score(y_test, y_pred_single)

# --- Results Analysis ---
print("\n" + "="*35)
print("FINAL PERFORMANCE COMPARISON")
print("="*35)
print(f"Single Decision Tree Accuracy: {single_accuracy:.4f}")
print(f"Bagging Ensemble Accuracy:      {bagging_accuracy:.4f}")

improvement = bagging_accuracy - single_accuracy
if improvement > 0:
    print(f"Result: Bagging improved accuracy by {improvement:.4f}!")
elif improvement == 0:
    print("Result: Both models performed equally on this simple dataset.")
else:
    print("Result: On this small split, the single tree held up well.")
