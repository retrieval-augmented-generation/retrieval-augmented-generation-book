# tune_ef_search.py
import faiss
import numpy as np
import time

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


# Build a single index with fixed M and ef_construction
M = 32
ef_construction = 200
index = faiss.IndexHNSWFlat(dimension, M)
index.hnsw.efConstruction = ef_construction
index.add(embeddings)

# Test different ef_search values
ef_values = [16, 32, 64, 128, 256, 512]
results = []

for ef in ef_values:
    index.hnsw.efSearch = ef

    start = time.perf_counter()
    for _ in range(3):  # Average over 3 runs
        distances, indices = index.search(queries, 10)
    elapsed = (time.perf_counter() - start) / 3 * 1000

    recall = recall_at_k(indices_flat, indices, k=10)
    results.append((ef, recall, elapsed / num_queries))

print(f"{'ef_search':<12} {'Recall@10':<12} {'Latency/query (ms)'}")
print("-" * 44)
for ef, recall, latency in results:
    print(f"{ef:<12} {recall:<12.3f} {latency:.2f}")
