import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Task 1: Train a Random Forest ---
print("Task 1: Loading and Splitting Dataset...")

# 1.1 Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 1.2 Split the Dataset (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1.3 Train the Random Forest Model
# n_estimators=100 means we are building a forest of 100 individual trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("✓ Random Forest model trained successfully.")

# --- Task 2: Evaluate the Model Performance ---
print("\nTask 2: Evaluating Model...")

# 2.1 Make Predictions
y_pred = rf_model.predict(X_test)

# 2.2 Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Task 3: Explore Feature Importances ---
print("\nTask 3: Extracting Feature Importance...")

# 3.1 Extract and format the data
importances = rf_model.feature_importances_
feature_names = iris.feature_names
feature_importance_df = pd.DataFrame({
    'feature': feature_names, 
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\nFeature Importance Scores:")
print(feature_importance_df)

# 3.2 Visualize Feature Importances
# Note: In an Ubuntu terminal, this saves the plot as a file
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Random Forest: Feature Importance Analysis')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')

# Save the plot since we are in a terminal environment
plt.savefig('feature_importance_plot.png')
print("\n✓ Success: Visualization saved as 'feature_importance_plot.png'")
