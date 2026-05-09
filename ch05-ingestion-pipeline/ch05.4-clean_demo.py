# clean_demo.py -- Demonstrate text cleaning before and after
from text_cleaner import clean_text

raw_text = """
\u00a0\u00a0 Annual  Report   2024\n\n\n\n\n
Page 1 of 47

CONFIDENTIAL

The company\u2019s   revenue grew   by 15%
compared to\u00a0the\u00a0previous fiscal year.


Page 2 of 47



Revenue breakdown by   segment:
\u2022 Enterprise: $4.2B
\u2022 Consumer: $1.8B


Page 3 of 47
"""

print("=== BEFORE CLEANING ===")
print(repr(raw_text[:200]))
print(f"\nLength: {len(raw_text)} chars")

cleaned = clean_text(raw_text)

print("\n=== AFTER CLEANING ===")
print(cleaned)
print(f"\nLength: {len(cleaned)} chars")
