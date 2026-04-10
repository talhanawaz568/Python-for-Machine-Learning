import matplotlib
# Use Agg backend for headless Ubuntu environment
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

# Set a random seed for consistent results
np.random.seed(42)

# --- Task 1: Create a Histogram for Distribution ---
print("Task 1: Generating Histogram...")
# Generating 1000 points from a normal distribution (Bell Curve)
data = np.random.normal(0, 1, 1000)

plt.figure(figsize=(10, 6))
# bins=30 divides the data range into 30 bars
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of Normally Distributed Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig('histogram_plot.png')
print("✓ Histogram saved as 'histogram_plot.png'")


# --- Task 2 & 3: Scatter Plot with Annotation ---
print("\nTask 2 & 3: Generating Annotated Scatter Plot...")
# x is random values, y is x plus some noise (shows a positive correlation)
x = np.random.rand(50)
y = x + np.random.normal(0, 0.1, 50)

plt.figure(figsize=(10, 6))
# c='red' sets color, marker='o' sets the point shape
plt.scatter(x, y, c='red', marker='o', alpha=0.6, label='Data Points')

# Task 3: Adding Annotation
# We pick one specific point to highlight (e.g., the point with the highest x value)
max_idx = np.argmax(x)
plt.annotate('Highest Value', 
             xy=(x[max_idx], y[max_idx]), 
             xytext=(x[max_idx]-0.2, y[max_idx]+0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

plt.title('Scatter Plot with Trend Annotation')
plt.xlabel('Study Hours (Scaled)')
plt.ylabel('Test Scores (Scaled)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)

plt.savefig('annotated_scatter.png')
print("✓ Annotated scatter plot saved as 'annotated_scatter.png'")
