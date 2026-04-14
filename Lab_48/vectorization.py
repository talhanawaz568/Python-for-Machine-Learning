import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# --- Task 1: Use CountVectorizer ---
print("Task 1: Running CountVectorizer...")

# 1.2 Define Sample Text Data
documents = [
    "Text analysis is an interesting field.",
    "Machine Learning is part of data science.",
    "Text analysis involves understanding data."
]

# 1.3 Initialize and Fit CountVectorizer
# This creates a "Bag of Words" model
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(documents)

# 1.4 Display Vocabulary and Counts
feature_names = vectorizer.get_feature_names_out()
print("\nVocabulary (Features):")
print(feature_names)

print("\nVectorized Data (Word Counts):")
count_df = pd.DataFrame(count_matrix.toarray(), columns=feature_names)
print(count_df)

# --- Task 2: Apply TF-IDF Transformation ---
print("\nTask 2: Computing TF-IDF Values...")

# 2.2 Compute TF-IDF
# TF = Term Frequency (How often it appears in THIS document)
# IDF = Inverse Document Frequency (How unique it is to this document)
tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

# 2.3 Display TF-IDF Results
print("\nTF-IDF Matrix (Weighted Importance):")
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
print(tfidf_df.round(4)) # Rounding for cleaner output

# --- Task 3: Compare Vectorized Outputs ---
print("\n--- Task 3: Comparison Analysis ---")

# Let's look at the word "is"
print(f"Count of 'is':\n{count_df['is'].values}")
print(f"TF-IDF of 'is':\n{tfidf_df['is'].values}")

print("\nOBSERVATION:")
print("Notice that words like 'is', which appear in multiple documents, ")
print("have their weight reduced in the TF-IDF matrix compared to words ")
print("that are unique to a single document.")
