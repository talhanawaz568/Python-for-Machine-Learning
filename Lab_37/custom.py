import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# --- Task 1 & 2: Define the Custom Transformer Class ---

class MyCustomTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that adds a constant value to the dataset.
    Implementing BaseEstimator and TransformerMixin allows it to 
    work seamlessly with Pipelines and GridSearch.
    """
    def __init__(self, parameter=1):
        # We store our configuration parameters here
        self.parameter = parameter

    def fit(self, X, y=None):
        # The fit method is used to calculate stats (like mean or max).
        # For this simple transformer, we just return self.
        print("Fitting the Custom Transformer...")
        return self

    def transform(self, X):
        # The transform method applies the logic.
        # We ensure it returns a copy or a new object to avoid modifying original data.
        print(f"Transforming data: Adding {self.parameter} to all values.")
        return X + self.parameter

# --- Task 3: Test the Transformer within a Pipeline ---

print("--- Step 1: Creating Sample Dataset ---")
data = {
    'feature1': [10, 20, 30],
    'feature2': [1, 2, 3]
}
X = pd.DataFrame(data)
print("Original Data:")
print(X)

print("\n--- Step 2: Building and Running the Pipeline ---")
# We integrate our custom class into a standard Pipeline
my_pipeline = Pipeline(steps=[
    ('custom_transform', MyCustomTransformer(parameter=5)),
])

# fit_transform calls fit() then transform() in one go
transformed_data = my_pipeline.fit_transform(X)

print("\n--- Step 3: Evaluation ---")
print("Transformed Data (Expected: all values + 5):")
print(transformed_data)

# Verification check
if (transformed_data.iloc[0,0] == X.iloc[0,0] + 5):
    print("\n✓ Success: The custom transformer integrated correctly!")
