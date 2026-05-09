# qdrant_setup.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)
import numpy as np
import time

# Load test data (run generate_test_embeddings.py first)
embeddings = np.load("ch04-vector-storage/embeddings.npy")
dimension = embeddings.shape[1]

client = QdrantClient(host="localhost", port=6333)

# Create a collection
if client.collection_exists("documents"):
    client.delete_collection("documents")
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=dimension,
        distance=Distance.COSINE
    )
)

# Load chunk metadata from the Acme Corp corpus
chunk_metadata = np.load("ch04-vector-storage/chunk_metadata.npy", allow_pickle=True)
num_real = len(chunk_metadata)

def source_type(filename):
    if filename.endswith(".pdf"): return "pdf"
    if filename.endswith(".html"): return "html"
    if filename.endswith(".docx"): return "docx"
    return "markdown"

points = []
for i, vec in enumerate(embeddings):
    meta = chunk_metadata[i % num_real]
    points.append(PointStruct(
        id=i,
        vector=vec.tolist(),
        payload={
            "source": source_type(meta["source"]),
            "filename": meta["source"],
            "chunk_index": meta["chunk_index"],
        }
    ))

# Insert in batches
batch_size = 1000
start_insert = time.perf_counter()
for i in range(0, len(points), batch_size):
    client.upsert(
        collection_name="documents",
        points=points[i:i + batch_size]
    )
insert_time = time.perf_counter() - start_insert
print(f"Inserted {len(points)} points in {insert_time:.1f}s")
