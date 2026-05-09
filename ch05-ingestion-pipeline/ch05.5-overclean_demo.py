# overclean_demo.py -- Demonstrate the danger of over-cleaning
from text_cleaner import remove_empty_sections

# This text has short paragraphs that look like noise but are real data
technical_text = """Configuration steps:
1. Set the connection pool size to 20
2. Enable SSL verification
3. Set timeout to 30 seconds

Connection pool values:

10

20

50"""

print("=== ORIGINAL ===")
print(technical_text.strip())

# Aggressive cleaning with a high min_chars threshold
over_cleaned = remove_empty_sections(technical_text, min_chars=25)

print("\n=== AFTER OVER-CLEANING (min_chars=25) ===")
print(over_cleaned.strip())
