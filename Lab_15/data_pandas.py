import pandas as pd
import matplotlib
# Use Agg backend for headless Ubuntu environment
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- Task 1: Generate Summary Statistics ---
print("Task 1: Loading and Examining Dataset...")
df = pd.read_csv('student_performance.csv')

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Descriptive Statistics ---")
# describe() gives us mean, min, max, and quartiles
print(df.describe())

# --- Task 2: Identify Trends and Patterns ---
print("\nTask 2: Finding Patterns...")

# 2.1 Missing Data
print("\nMissing values check:")
print(df.isnull().sum())

# Fill missing study hours with the mean so the correlation works
df['Study_Hours'] = df['Study_Hours'].fillna(df['Study_Hours'].mean())

# 2.2 Correlation Analysis
# This tells us if more study hours lead to higher exam scores
print("\nCorrelation Matrix:")
corr = df.corr()
print(corr)

# --- Task 3: Visualize Distributions ---
print("\nTask 3: Generating Visualizations...")

# Histogram for Exam Scores
plt.figure(figsize=(8, 5))
df['Exam_Score'].hist(bins=5, color='skyblue', edgecolor='black')
plt.title('Distribution of Exam Scores')
plt.xlabel('Score')
plt.ylabel('Number of Students')
plt.savefig('exam_score_hist.png')
print("✓ Histogram saved as 'exam_score_hist.png'")

# Box Plot for Study vs Sleep
plt.figure(figsize=(8, 5))
df.boxplot(column=['Study_Hours', 'Sleep_Hours'])
plt.title('Comparison of Study vs Sleep Hours')
plt.savefig('study_sleep_boxplot.png')
print("✓ Boxplot saved as 'study_sleep_boxplot.png'")

print("\n" + "="*40)
print("EDA CONCLUSION")
print("="*40)
print(f"The correlation between Study_Hours and Exam_Score is: {corr.loc['Study_Hours', 'Exam_Score']:.2f}")
print("A value close to 1.0 means a very strong positive relationship!")


#CSV Data
#echo "Student_ID,Study_Hours,Sleep_Hours,Exam_Score" > student_performance.csv
echo "1,10,7,85" >> student_performance.csv
echo "2,15,6,92" >> student_performance.csv
echo "3,5,8,60" >> student_performance.csv
echo "4,8,7,75" >> student_performance.csv
echo "5,12,6,88" >> student_performance.csv
echo "6,2,9,45" >> student_performance.csv
echo "7,14,5,95" >> student_performance.csv
echo "8,7,7,70" >> student_performance.csv
echo "9,,8,65" >> student_performance.csv
echo "10,11,6,82" >> student_performance.csv
