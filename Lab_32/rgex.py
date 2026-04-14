import re

# --- Task 1: Basic Pattern Matching ---
print("--- Task 1: Simple Pattern Matching ---")
text_1 = "My phone number is 123-456-7890."
# Pattern: 3 digits, hyphen, 3 digits, hyphen, 4 digits
pattern_1 = r"\d{3}-\d{3}-\d{4}"

match = re.search(pattern_1, text_1)
if match:
    print(f"✓ Found Phone Number: {match.group()}")

# --- Task 2: Finding Multiple Patterns ---
print("\n--- Task 2: Extracting Emails ---")
text_2 = "Contact us at info@example.com or support@example.org."
# Pattern: matches alphanumeric chars + special chars, @, domain, and TLD
pattern_2 = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

emails = re.findall(pattern_2, text_2)
print(f"✓ Found Emails: {emails}")

# --- Task 3: Data Cleaning & Normalization ---
print("\n--- Task 3: Cleaning and Normalizing Data ---")

# 3.1 Removing Punctuation
dirty_text = "Hello! This, is a sample text. It's meant for cleaning."
# Pattern: [^...] means 'not'. So this matches anything NOT a word or space.
punctuation_pattern = r"[^\w\s]"
cleaned_text = re.sub(punctuation_pattern, "", dirty_text)
print(f"✓ Cleaned Text: {cleaned_text}")

# 3.2 Standardizing Phone Numbers
# This is a common real-world DevOps/Data problem
raw_numbers = ["1234567890", "123-456-7890", "(123) 456-7890", "123 456 7890"]
# Pattern: Groups digits into three parts while ignoring spaces, hyphens, or brackets
clean_pattern = r".*?(\d{3}).*?(\d{3}).*?(\d{4})"

standardized = []
for num in raw_numbers:
    # re.sub uses \1, \2, \3 to refer to the groups inside () in the pattern
    standardized.append(re.sub(clean_pattern, r"(\1) \2-\3", num))

print("Standardized Numbers:")
for s_num in standardized:
    print(f"  - {s_num}")

print("\n--- Lab Conclusion ---")
print("Regex successfully applied for extraction, cleaning, and formatting.")
