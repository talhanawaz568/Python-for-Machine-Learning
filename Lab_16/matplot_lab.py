import matplotlib
# CRITICAL: Use 'Agg' backend for Ubuntu terminal environments 
# This allows saving files without needing a graphical window.
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import numpy as np

# --- Task 1: Plot a Line Graph ---
print("Task 1: Generating Line Graph (Sine Wave)...")
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, color='blue', label='Sine Wave')

# --- Task 3: Customization (Part 1) ---
plt.title('Sine Wave Trend')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.savefig('line_graph.png')
print("✓ Line graph saved as 'line_graph.png'")


# --- Task 2: Create a Bar Chart ---
print("\nTask 2: Generating Bar Chart...")
categories = ['A', 'B', 'C', 'D']
values = [4, 7, 1, 8]

plt.figure(figsize=(8, 4))
plt.bar(categories, values, color='green')

# --- Task 3: Customization (Part 2) ---
plt.title('Category Comparison')
plt.xlabel('Categories')
plt.ylabel('Values')

plt.savefig('bar_chart.png')
print("✓ Bar chart saved as 'bar_chart.png'")


# --- Bonus: Scatter Plot ---
print("\nBonus: Generating Scatter Plot...")
random_x = np.random.rand(50)
random_y = np.random.rand(50)

plt.figure(figsize=(8, 4))
plt.scatter(random_x, random_y, alpha=0.5, color='red')
plt.title('Random Distribution')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')

plt.savefig('scatter_plot.png')
print("✓ Scatter plot saved as 'scatter_plot.png'")
