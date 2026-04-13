import pandas as pd
import numpy as np
import matplotlib
# Use Agg backend for Ubuntu terminal compatibility
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- Task 1: Load a Dataset ---
print("Step 1: Loading Iris Dataset...")
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

print("Dataset Preview:")
print(data.head())

# --- Task 2: Train a k-NN Classifier ---
print("\nStep 2: Splitting Data and Training Initial Model (k=3)...")
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42
)

# Initialize and Train
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, y_train)

# Evaluate
predictions_3 = knn_3.predict(X_test)
accuracy_3 = accuracy_score(y_test, predictions_3)
print(f"Accuracy with k=3: {accuracy_3:.4f}")

# --- Task 3: Experiment with Different k Values ---
print("\nStep 3: Experimenting with k values 1 through 10...")
accuracies = []
k_range = range(1, 11)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k = {k:2} | Accuracy: {acc:.4f}")

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o', linestyle='--', color='purple')
plt.xlabel('Value of k (Number of Neighbors)')
plt.ylabel('Model Accuracy')
plt.title('k-NN Performance: Varying k Neighbors')
plt.xticks(k_range)
plt.grid(True)

# Save the plot
plt.savefig('knn_performance.png')
print("\n✓ Performance plot saved as 'knn_performance.png'")

# --- Lab Analysis ---
best_k = k_range[np.argmax(accuracies)]
print("-" * 40)
print(f"ANALYSIS COMPLETE")
print(f"Best performance observed at k = {best_k}")
print("-" * 40)
