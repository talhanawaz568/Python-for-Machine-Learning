import requests
from bs4 import BeautifulSoup

# --- Task 1: Verify Libraries ---
print("--- Task 1: Library Verification ---")
try:
    import requests
    from bs4 import BeautifulSoup
    print("Libraries imported successfully!")
except ImportError as e:
    print(f"Error importing libraries: {e}")

# --- Task 2: Scrape Sample Data ---
print("\n--- Task 2: Fetching Webpage Content ---")

# Step 1 & 2: Define URL and fetch content
url = "https://example.com"
try:
    response = requests.get(url, timeout=10)

    # Verify the response status (200 means Success)
    if response.status_code == 200:
        print(f"Successfully fetched content from {url}!")
        
        # Step 3: Parse the HTML Content
        soup = BeautifulSoup(response.content, "html.parser")
        
        print("\nPrettified HTML Preview (First 300 chars):")
        print("-" * 30)
        print(soup.prettify()[:300])
        print("-" * 30)
        
        # --- Task 3: Parse and Clean Data ---
        print("\n--- Task 3: Data Extraction & Cleaning ---")
        
        # Step 1: Identify and Extract Specific Data
        # We will look for headers (h1) and paragraphs (p)
        headers = soup.find_all('h1')
        paragraphs = soup.find_all('p')
        
        print("Raw headers found:")
        for h in headers:
            print(f"  - {h}")
            
        # Step 2: Clean the Extracted Data
        # strip() removes leading/trailing whitespace and newlines
        cleaned_headers = [header.get_text().strip() for header in headers]
        cleaned_paragraphs = [p.get_text().strip() for p in paragraphs]
        
        print("\nCleaned Data for ML:")
        print(f"Headers: {cleaned_headers}")
        print(f"First Paragraph: {cleaned_paragraphs[0] if cleaned_paragraphs else 'None'}")

    else:
        print(f"Failed to fetch content. Status Code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the request: {e}")

print("\n--- Lab Conclusion ---")
print("Data is now in a clean list format, ready for NLP or ML processing.")
