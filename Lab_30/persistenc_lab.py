import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Task 1: Train a Simple Model ---
print("Step 1: Training the model...")
data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate initial accuracy for comparison later
initial_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained. Initial Accuracy: {initial_accuracy:.4f}")

# --- Task 2: Save the Model Using Joblib ---
print("\nStep 2: Saving the model to disk ('random_forest_model.pkl')...")
# Joblib is generally faster than the standard 'pickle' for large NumPy arrays
joblib.dump(model, 'random_forest_model.pkl')
print("✓ Model saved successfully.")

# --- Task 3: Reload the Model and Make a Prediction ---
print("\nStep 3: Reloading the model from the file...")
# Imagine this part of the code is running on a different machine or a month later
loaded_model = joblib.load('random_forest_model.pkl')

# Make predictions with the reloaded model
predictions = loaded_model.predict(X_test)
reloaded_accuracy = accuracy_score(y_test, predictions)

print("-" * 40)
print(f"Predicted Labels: {predictions}")
print(f"Actual Labels:    {y_test}")
print("-" * 40)
print(f"Reloaded Model Accuracy: {reloaded_accuracy:.4f}")

if initial_accuracy == reloaded_accuracy:
    print("SUCCESS: The persisted model performs exactly like the original!")
