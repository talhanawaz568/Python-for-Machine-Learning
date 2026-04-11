import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for Ubuntu terminal compatibility
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Task 1: Make Predictions on Test Data ---
print("Task 1: Loading and Splitting Dataset...")
# Load dataset
data = pd.read_csv('boston_housing.csv')
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)
print("✓ Predictions generated.")

# --- Task 2: Calculate MSE and R² Score ---
print("\nTask 2: Calculating Metrics...")

# 2.1 Mean Squared Error (MSE)
# This tells us the 'cost' of our errors
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# 2.2 R² Score
# This tells us the 'percentage' of accuracy
r2 = r2_score(y_test, y_pred)
print(f"R² Score:           {r2:.4f}")

# --- Task 3: Plot Predictions vs. Actual Values ---
print("\nTask 3: Visualizing Results...")
plt.figure(figsize=(8, 6))

# Scatter plot of actual vs predicted
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')

# Red Line: Where the dots SHOULD be if the model was 100% perfect
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Fit')

plt.title('Actual vs Predicted Values (Evaluation)')
plt.xlabel('Actual MEDV (Target)')
plt.ylabel('Predicted MEDV')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('regression_evaluation.png')
print("✓ Plot saved as 'regression_evaluation.png'")

# --- Lab Summary ---
print("\n" + "="*35)
print("INTERPRETATION GUIDE")
print("="*35)
print(f"The MSE of {mse:.2f} means that on average, our prediction squared is off by this much.")
print(f"The R² of {r2*100:.1f}% means our model explains most of the data's behavior.")
print("="*35)
