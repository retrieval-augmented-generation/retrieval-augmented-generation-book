from sentence_transformers import SentenceTransformer

from chunkers import chunk_semantic

sample_document = """
Data Retention Policy

1. General Requirements
All customer data must be retained for a minimum of 7 years from the date
of the last transaction. This applies to all data categories listed in
section 4.2, with the exception of anonymized datasets processed after the
effective date of this policy (January 1, 2024).

2. EU-Specific Requirements
For customers subject to GDPR, data retention is further constrained by the
principle of storage limitation. Personal data may only be retained for as
long as necessary for the purpose for which it was collected. The 7-year
general requirement applies only to transaction records; personal
identification data must be deleted or anonymized within 3 years unless a
specific legal basis for longer retention exists.

3. Data Deletion Procedures
Upon expiration of the retention period, data must be permanently deleted
within 90 calendar days. Deletion must be verified by the data protection
officer and logged in the compliance audit trail. Backup copies must be
purged within 180 days of the primary deletion.
""".strip()

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks, similarities = chunk_semantic(
    sample_document, model, threshold=0.3
)

print("Sentence-to-sentence similarities:")
for i, sim in enumerate(similarities):
    marker = " <-- SPLIT" if sim < 0.3 else ""
    print(f"  Sentence {i} -> {i+1}: {sim:.3f}{marker}")

print(f"\nResulting chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
