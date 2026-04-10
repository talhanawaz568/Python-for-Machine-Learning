import seaborn as sns
import pandas as pd
import matplotlib
# Use Agg backend for headless Ubuntu environment
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- Task 1: Setup and Theming ---
print("Task 1: Setting up Seaborn theme and palette...")
# Setting a theme makes the plots look modern and clean automatically
sns.set_theme(style="whitegrid")
sns.set_palette('pastel')

# --- Task 2.1: Load Sample Dataset ---
print("Task 2: Loading 'tips' dataset...")
# This dataset contains info on total bills, tips, day of week, etc.
tips = sns.load_dataset('tips')

# --- Task 2.2 & 3.3: Create a Box Plot ---
print("Generating Box Plot...")
plt.figure(figsize=(10, 6))
# Seaborn automatically groups data by 'day' and calculates statistics
sns.boxplot(x='day', y='total_bill', data=tips)

plt.title('Customized Box Plot of Total Bill by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill ($)')

plt.savefig('seaborn_boxplot.png')
print("✓ Box plot saved as 'seaborn_boxplot.png'")

# --- Task 2.3: Create a Violin Plot ---
print("Generating Violin Plot...")
plt.figure(figsize=(10, 6))
# inner='quartile' draws the 25th, 50th, and 75th percentiles inside the violin
sns.violinplot(x='day', y='total_bill', data=tips, inner='quartile')

plt.title('Violin Plot of Total Bill by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill ($)')

plt.savefig('seaborn_violinplot.png')
print("✓ Violin plot saved as 'seaborn_violinplot.png'")

# --- Bonus: Comparison Analysis ---
print("\n" + "="*40)
print("LAB ANALYSIS")
print("="*40)
print("1. Box Plot: Best for seeing outliers (the dots outside the whiskers).")
print("2. Violin Plot: Best for seeing the 'density' or shape of the data.")
print("   The wider sections of the 'violin' represent where most bills occur.")
