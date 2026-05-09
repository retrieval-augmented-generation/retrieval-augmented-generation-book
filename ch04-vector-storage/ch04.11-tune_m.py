# tune_m.py
import faiss
import numpy as np
import time
import os

# Load test data (run generate_test_embeddings.py first)
embeddings = np.load("ch04-vector-storage/embeddings.npy")
queries = np.load("ch04-vector-storage/queries.npy")
num_queries, dimension = queries.shape

# Ground truth from flat index
index_flat = faiss.IndexFlatIP(dimension)
index_flat.add(embeddings)
_, indices_flat = index_flat.search(queries, 10)

def recall_at_k(ground_truth, predictions, k):
    recalls = []
    for gt, pred in zip(ground_truth, predictions):
        recalls.append(len(set(gt[:k]) & set(pred[:k])) / k)
    return np.mean(recalls)


M_values = [8, 16, 32, 48, 64]
ef_search = 64  # Fixed
ef_construction = 200  # Fixed
results = []

for M in M_values:
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)
    index.hnsw.efSearch = ef_search

    # Measure memory by writing to disk
    faiss.write_index(index, "/tmp/test_index.bin")
    mem_mb = os.path.getsize("/tmp/test_index.bin") / (1024 * 1024)

    start = time.perf_counter()
    distances, indices = index.search(queries, 10)
    elapsed = (time.perf_counter() - start) * 1000

    recall = recall_at_k(indices_flat, indices, k=10)
    results.append((M, recall, elapsed / num_queries, mem_mb))

print(f"{'M':<6} {'Recall@10':<12} {'Latency (ms)':<15} {'Memory (MB)'}")
print("-" * 49)
for M, recall, latency, mem in results:
    print(f"{M:<6} {recall:<12.3f} {latency:<15.2f} {mem:.0f}")
