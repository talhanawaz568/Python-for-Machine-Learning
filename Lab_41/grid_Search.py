import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# --- Task 1: Define Data and Parameter Grid ---
print("Task 1: Loading Dataset and Defining Parameters...")

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Support Vector Classifier
model = SVC()

# Define the Parameter Grid
# C: Controls the trade-off between smooth boundary and classifying training points correctly
# Gamma: Defines how far the influence of a single training example reaches
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# --- Task 2: Run GridSearchCV ---
print("\nTask 2: Starting Grid Search...")
# cv=5: Uses 5-fold cross-validation
# n_jobs=-1: Uses all available CPU cores to speed up the search
grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy', 
    verbose=1, 
    n_jobs=-1
)

# Fit the search to the training data
grid_search.fit(X_train, y_train)

# --- Task 3: Evaluate and Select the Best Parameters ---
print("\nTask 3: Final Results and Evaluation")
print("-" * 40)

# Display the winning combination
print(f"Best Hyperparameters Found: {grid_search.best_params_}")
print(f"Best Training (CV) Accuracy: {grid_search.best_score_:.4f}")

# Evaluate the 'best_estimator_' on the unseen test data
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)

print(f"Final Test Set Accuracy:      {test_accuracy:.4f}")
print("-" * 40)

# Detailed Report
y_pred = best_model.predict(X_test)
print("\nDetailed Classification Report for Best Model:")
print(classification_report(y_test, y_pred))
