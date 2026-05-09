from chunkers import chunk_markdown

markdown_doc = """## Data Retention Policy

All customer data must be retained for a minimum of 7 years.

## General Requirements

Retention applies to all data categories in section 4.2.
Anonymized datasets are excluded after January 1, 2024.

## EU-Specific Requirements

GDPR constrains retention via storage limitation principle.
Personal ID data must be deleted within 3 years.

## Data Deletion Procedures

Data must be permanently deleted within 90 calendar days
of retention period expiration.
"""

chunks = chunk_markdown(markdown_doc)
for chunk in chunks:
    print(f"[{chunk['header']}] ({chunk['char_count']} chars)")
    print(chunk["text"][:100] + "...")
    print()
