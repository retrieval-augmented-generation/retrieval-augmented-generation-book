# benchmark_qdrant.py
from qdrant_client import QdrantClient
import numpy as np
import faiss
import time

# Load test data
embeddings = np.load("ch04-vector-storage/embeddings.npy")
queries = np.load("ch04-vector-storage/queries.npy")
dimension = embeddings.shape[1]

# Get ground truth from FAISS flat index
index_flat = faiss.IndexFlatIP(dimension)
index_flat.add(embeddings)
_, indices_flat = index_flat.search(queries, 10)

def recall_at_k(ground_truth, predictions, k):
    recalls = []
    for gt, pred in zip(ground_truth, predictions):
        gt_set = set(gt[:k])
        pred_set = set(pred[:k])
        recalls.append(len(gt_set & pred_set) / k)
    return np.mean(recalls)

client = QdrantClient(host="localhost", port=6333)

latencies = []
all_qdrant_ids = []

for q in queries:
    start = time.perf_counter()
    results = client.query_points(
        collection_name="documents",
        query=q.tolist(),
        limit=10
    )
    latencies.append((time.perf_counter() - start) * 1000)
    all_qdrant_ids.append([r.id for r in results.points])

avg_latency = np.mean(latencies)
p99_latency = np.percentile(latencies, 99)
recall_qdrant = recall_at_k(indices_flat, np.array(all_qdrant_ids), k=10)

print(f"Qdrant: recall@10={recall_qdrant:.3f}, "
      f"avg={avg_latency:.1f}ms, p99={p99_latency:.1f}ms")
