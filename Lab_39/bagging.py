import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Task 1: Load and Prepare Data ---
print("Task 1: Loading Iris Dataset...")
iris = load_iris()
X, y = iris.data, iris.target

print(f"Features: {iris.feature_names}")
print(f"Number of classes: {len(np.unique(y))}")

# Split the dataset (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Task 2: Train and Evaluate AdaBoost ---
print("\nTask 2: Training AdaBoost Classifier...")
# n_estimators=50 means 50 weak learners (stumps) will be trained sequentially
boosting_model = AdaBoostClassifier(n_estimators=50, random_state=42)
boosting_model.fit(X_train, y_train)

y_pred_boost = boosting_model.predict(X_test)
accuracy_boost = accuracy_score(y_test, y_pred_boost)
print(f"Accuracy of AdaBoost model: {accuracy_boost:.4f}")

# --- Task 3: Comparison with Other Methods ---
print("\nTask 3: Running Comparisons...")

# 3.1 Bagging Comparison
bagging_model = BaggingClassifier(n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)
y_pred_bag = bagging_model.predict(X_test)
accuracy_bag = accuracy_score(y_test, y_pred_bag)
print(f"Accuracy of Bagging model:  {accuracy_bag:.4f}")

# 3.2 Stacking Comparison
# Stacking uses a 'final_estimator' to learn how to best combine the other models
estimators = [
    ('bagging', BaggingClassifier(n_estimators=10, random_state=42)),
    ('boosting', AdaBoostClassifier(n_estimators=10, random_state=42))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(X_train, y_train)
y_pred_stack = stacking_model.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack)
print(f"Accuracy of Stacking model: {accuracy_stack:.4f}")

# Final Summary
print("\n" + "="*35)
print("FINAL RESULTS SUMMARY")
print("="*35)
results = pd.DataFrame({
    'Method': ['AdaBoost', 'Bagging', 'Stacking'],
    'Accuracy': [accuracy_boost, accuracy_bag, accuracy_stack]
})
print(results.to_string(index=False))
