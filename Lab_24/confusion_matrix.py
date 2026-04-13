import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for Ubuntu terminal compatibility
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

# --- Task 1: Generate Predictions ---
print("Step 1: Loading Iris Dataset and Training Random Forest...")
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset (30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# --- Task 2: Create and Visualize Confusion Matrix ---
print("\nStep 2: Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

# Visualize using Seaborn Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix: Iris Species Prediction')
plt.savefig('confusion_matrix.png')
print("✓ Confusion matrix heatmap saved as 'confusion_matrix.png'")

# --- Task 3: Calculate Evaluation Metrics ---
print("\nStep 3: Calculating Advanced Metrics...")

# We use average='weighted' because Iris has 3 classes (multi-class)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("-" * 35)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("-" * 35)

# Interpretation Guide
print("\nQuick Interpretation:")
print("- Diagonal values (top-left to bottom-right) are CORRECT predictions.")
print("- Off-diagonal values are MISTAKES (misclassifications).")
