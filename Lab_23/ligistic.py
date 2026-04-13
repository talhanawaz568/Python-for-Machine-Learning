import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for Ubuntu terminal
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Task 1: Load a Binary Classification Dataset ---
print("Step 1: Loading Iris dataset for Binary Classification...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Filter out the 3rd class (index 2) to make it a Binary problem
# Now we are only predicting between 'Setosa' and 'Versicolour'
X_binary = X[y != 2]
y_binary = y[y != 2]

# --- Task 2: Train a Logistic Regression Model ---
print("Step 2: Splitting and Training...")
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

# Initialize and Train
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Task 3: Evaluate Model Accuracy ---
print("Step 3: Making Predictions and Evaluating...")
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

# --- Detailed Analysis ---
print("\n" + "="*35)
print(f"RESULTS FOR BINARY CLASSIFICATION")
print("="*35)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 35)

# Confusion Matrix: Shows where the model got it right/wrong
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolour']))
print("="*35)
