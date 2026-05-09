# tune_ef_construction.py
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


ef_construction_values = [50, 100, 200, 400, 800]
M = 32  # Fixed
ef_search = 64  # Fixed
results = []

for ef_c in ef_construction_values:
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = ef_c

    start_build = time.perf_counter()
    index.add(embeddings)
    build_time = time.perf_counter() - start_build

    index.hnsw.efSearch = ef_search
    distances, indices = index.search(queries, 10)
    recall = recall_at_k(indices_flat, indices, k=10)
    results.append((ef_c, recall, build_time))

print(f"{'ef_construction':<18} {'Recall@10':<12} {'Build time (s)'}")
print("-" * 42)
for ef_c, recall, bt in results:
    print(f"{ef_c:<18} {recall:<12.3f} {bt:.1f}")
