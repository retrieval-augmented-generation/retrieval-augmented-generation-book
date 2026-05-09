# comparative_retrieval_test.py
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from chunkers import (
    add_context_to_chunk,
    chunk_fixed_size,
    chunk_markdown,
    chunk_recursive,
    chunk_semantic,
)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()

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

def embed_chunks(chunks):
    """Embed a list of chunk texts."""
    texts = [c if isinstance(c, str) else c["text"]
             for c in chunks]
    return model.encode(texts)

def search(query, chunk_texts, chunk_embeddings, top_k=5):
    """Brute-force cosine similarity search."""
    query_embedding = model.encode([query])[0]
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1)
        * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(chunk_texts[i], similarities[i])
            for i in top_indices]

# Define test queries with expected answers.
# The first three are easy: the answer chunk contains the exact terms.
# The last three are harder. They require context the chunk loses when
# boundaries cut mid-section (pronoun resolution, section disambiguation,
# or knowing which rule applies to whom).
test_queries = [
    {
        "query": "How long must EU customer personal "
                 "data be retained?",
        "expected_substring": "3 years",
    },
    {
        "query": "What happens to data after the "
                 "retention period expires?",
        "expected_substring": "90 calendar days",
    },
    {
        "query": "Are anonymized datasets subject to "
                 "the 7-year rule?",
        "expected_substring": "exception",
    },
    {
        "query": "What is the deadline for removing "
                 "backup copies of deleted data?",
        "expected_substring": "180 days",
    },
    {
        "query": "What types of records does the 7-year "
                 "rule actually apply to?",
        "expected_substring": "transaction records",
    },
    {
        "query": "Who is responsible for verifying that "
                 "data has been deleted?",
        "expected_substring": "data protection officer",
    },
]

# Chunk with each strategy. We use a small chunk_size (200) so fixed-size
# splits mid-sentence and recursive produces more, smaller chunks. With
# chunk_size near the document length, every strategy looks identical.
CHUNK_SIZE = 200
OVERLAP = 30

recursive_chunks = chunk_recursive(
    sample_document, chunk_size=CHUNK_SIZE, overlap=OVERLAP
)
contextual_chunks = [
    add_context_to_chunk(client, sample_document, chunk)
    for chunk in recursive_chunks
]

strategies = {
    "fixed_size": chunk_fixed_size(
        sample_document, chunk_size=CHUNK_SIZE, overlap=OVERLAP
    ),
    "recursive": recursive_chunks,
    "semantic": chunk_semantic(
        sample_document, model, threshold=0.75
    )[0],
    "structure": [c["text"] for c in chunk_markdown(
        sample_document
    )] if "##" in sample_document else chunk_recursive(
        sample_document, chunk_size=CHUNK_SIZE, overlap=OVERLAP
    ),
    "contextual": contextual_chunks,
}

# Run the comparison
results = {}
for strategy_name, chunks in strategies.items():
    texts = [c if isinstance(c, str) else c["text"]
             for c in chunks]
    embeddings = embed_chunks(texts)
    hits = 0
    for tq in test_queries:
        top_results = search(
            tq["query"], texts, embeddings, top_k=2
        )
        found = any(
            tq["expected_substring"].lower() in result[0].lower()
            for result in top_results
        )
        if found:
            hits += 1
    results[strategy_name] = hits

print("\n=== Retrieval Accuracy (correct in top-2) ===")
print(f"{'Strategy':<20} {'Hits':>5} / {len(test_queries)}")
print("-" * 35)
for strategy, hits in results.items():
    pct = hits / len(test_queries) * 100
    print(f"{strategy:<20} {hits:>5} / "
          f"{len(test_queries)}  ({pct:.0f}%)")
