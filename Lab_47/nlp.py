import nltk
import os

# --- Task 1: Setup and Downloads ---
print("Task 1: Initializing NLTK...")

# NLTK requires downloading specific datasets to function
try:
    nltk.download('punkt', quiet=True)     # For tokenization
    nltk.download('stopwords', quiet=True) # For common word removal
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# --- Task 2: Tokenization ---
print("\nTask 2: Tokenizing Text...")

from nltk.tokenize import sent_tokenize, word_tokenize

sample_text = ("Natural Language Processing (NLP) enables computers to understand "
               "and communicate in human language. It's an exciting field!")

# 2.2 Tokenizing into Sentences
sentences = sent_tokenize(sample_text)
print(f"Sentences Found: {len(sentences)}")
print(sentences)

# 2.3 Tokenizing into Words
words = word_tokenize(sample_text)
print(f"\nWords Found: {len(words)}")
print(words[:10], "...") # Show first 10 words

# --- Task 3: Preprocessing ---
print("\nTask 3: Text Preprocessing...")

# 3.2 Stopword Removal
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
# We convert to lowercase to ensure we catch all matches
filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]

print(f"Words remaining after stopword removal: {len(filtered_words)}")
print(filtered_words)

# 3.4 Stemming
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stemmed_words = [ps.stem(word) for word in filtered_words]

print("\nFinal Stemmed Words (Root forms):")
print(stemmed_words)

# --- Lab Summary ---
print("\n" + "="*40)
print("PREPROCESSING SUMMARY")
print("="*40)
print(f"Original word count: {len(words)}")
print(f"Cleaned word count:  {len(stemmed_words)}")
print("="*40)
