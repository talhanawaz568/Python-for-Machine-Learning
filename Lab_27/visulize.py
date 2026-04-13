import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz
import os

# --- Task 2.1: Load and Prepare Data ---
print("Step 1: Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# --- Task 2.2: Train the Decision Tree Model ---
print("Step 2: Training the Decision Tree...")
clf = DecisionTreeClassifier(random_state=42)
clf = clf.fit(X, y)

# --- Task 2.3: Export the Tree Visualization ---
print("Step 3: Generating Graphviz DOT data...")
# export_graphviz provides much more detail and formatting options than basic plots
dot_data = tree.export_graphviz(
    clf, 
    out_file=None, 
    feature_names=iris.feature_names, 
    class_names=list(iris.target_names), 
    filled=True,          # Colors the nodes based on their class
    rounded=True,         # Rounds the corners of the node boxes
    special_characters=True
)  

# Render the graph
graph = graphviz.Source(dot_data)  

# --- Saving the output ---
# On an Ubuntu terminal, we save it as a PDF or PNG to view later
output_filename = "iris_decision_tree"
graph.render(output_filename, format="png", cleanup=True)

print(f"✓ Visualization saved as '{output_filename}.png'")
