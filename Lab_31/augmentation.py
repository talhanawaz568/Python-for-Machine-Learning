import numpy as np
from PIL import Image
import nltk
from nltk.corpus import wordnet
import os

# Ensure NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- Task 2: Image Augmentation ---
print("--- Task 2: Image Augmentation ---")
if os.path.exists('sample_image.jpg'):
    image = Image.open('sample_image.jpg')
    print("✓ Successfully loaded sample_image.jpg")

    # Rotating 
    image.rotate(90).save('augmented_rotated.jpg')
    print("✓ Saved: augmented_rotated.jpg")

    # Flipping
    image.transpose(Image.FLIP_LEFT_RIGHT).save('augmented_flipped.jpg')
    print("✓ Saved: augmented_flipped.jpg")
else:
    print("Error: sample_image.jpg not found. Run the terminal command to create it first!")

# --- Task 3: Text Augmentation ---
print("\n--- Task 3: Text Augmentation ---")
text = "Data augmentation is a technique to expand dataset diversity."

def synonym_replacement(text, n):
    words = text.split()
    new_words = words.copy()
    count = 0
    
    for i in range(len(words)):
        if count >= n: break
        
        # WordNet works best with lowercase
        clean_word = words[i].lower().strip('.')
        synsets = wordnet.synsets(clean_word)
        
        if synsets:
            # Get a synonym that isn't the original word
            synonyms = [lemma.name() for syn in synsets for lemma in syn.lemmas() if lemma.name() != clean_word]
            if synonyms:
                new_words[i] = synonyms[0].replace('_', ' ')
                count += 1
                
    return ' '.join(new_words)

augmented_text = synonym_replacement(text, 2)
print(f"Original Text:  {text}")
print(f"Augmented Text: {augmented_text}")

print("\n--- Task 4: Evaluation Summary ---")
print("Augmentation complete. Check your folder for the new .jpg files.")
