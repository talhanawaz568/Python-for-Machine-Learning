import pandas as pd
import matplotlib
# Use Agg backend for Ubuntu terminal environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# --- Task 1: Load Dataset and Train Model ---
print("Step 1: Preparing Iris Dataset...")
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree (Using Gini by default)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# --- Task 2: Explore Tree Structure via Text ---
print("\nStep 2: Generating Decision Rules (Text Representation):")
tree_rules = export_text(clf, feature_names=list(iris['feature_names']))
print("-" * 30)
print(tree_rules)
print("-" * 30)

# --- Task 3: Split Criteria and Feature Importance ---
print("\nStep 3.1: Comparing Gini vs Entropy Criteria...")
# Retrain using entropy for comparison
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_entropy.fit(X_train, y_train)
print(f"Gini model depth: {clf.get_depth()}")
print(f"Entropy model depth: {clf_entropy.get_depth()}")

# Step 3.2: Determine Feature Importance
print("\nStep 3.2: Calculating Feature Importance...")
importance = clf.feature_importances_
feature_importance_df = pd.DataFrame(
    importance, 
    index=iris['feature_names'], 
    columns=["Importance"]
).sort_values(by="Importance", ascending=False)

print(feature_importance_df)

# Step 3.3: Visualizing the Tree
print("\nStep 3.3: Generating Tree Visualization...")
plt.figure(figsize=(20,10))
plot_tree(clf, 
          filled=True, 
          feature_names=iris['feature_names'], 
          class_names=list(iris['target_names']),
          rounded=True,
          fontsize=12)

plt.title("Decision Tree Visualization - Iris Dataset")
plt.savefig('decision_tree_visual.png')
print("✓ Tree diagram saved as 'decision_tree_visual.png'")
