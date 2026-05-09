# qdrant_query.py
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
import time

queries = np.load("ch04-vector-storage/queries.npy")
client = QdrantClient(host="localhost", port=6333)

# Query WITHOUT filter
start = time.perf_counter()
results_unfiltered = client.query_points(
    collection_name="documents",
    query=queries[0].tolist(),
    limit=10
)
unfiltered_time = (time.perf_counter() - start) * 1000

print(f"Unfiltered search: {unfiltered_time:.1f} ms")
for r in results_unfiltered.points[:3]:
    print(f"  ID={r.id}, score={r.score:.4f}, source={r.payload['source']}")

# Query WITH filter: only PDF sources
start = time.perf_counter()
results_filtered = client.query_points(
    collection_name="documents",
    query=queries[0].tolist(),
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value="pdf")
            )
        ]
    ),
    limit=10
)
filtered_time = (time.perf_counter() - start) * 1000

print(f"\nFiltered search (PDF only): {filtered_time:.1f} ms")
for r in results_filtered.points[:3]:
    print(f"  ID={r.id}, score={r.score:.4f}, source={r.payload['source']}")
