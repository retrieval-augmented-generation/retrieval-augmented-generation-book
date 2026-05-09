# verify_index.py - Test queries against the populated index
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sentence_transformers import SentenceTransformer
from database import get_connection

model = SentenceTransformer("all-MiniLM-L6-v2")

test_queries = [
    "What was the revenue growth last year?",
    "How do I authenticate API requests?",
    "What is the data retention policy?",
    "What happens during a service outage?",
    "What is the refund policy?",
]

conn = get_connection()

for query in test_queries:
    query_embedding = model.encode(query).tolist()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, source_file, chunk_index,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 3
            """,
            (query_embedding, query_embedding),
        )
        results = cur.fetchall()

    print(f"\nQuery: {query}")
    for content, source, idx, sim in results:
        print(f"  [{sim:.3f}] {source} (chunk {idx}): {content[:80]}...")

conn.close()
