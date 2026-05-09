# benchmark_faiss.py -- Compare all three FAISS index types
import faiss
import numpy as np
import time

def recall_at_k(ground_truth, predictions, k):
    """Compute average recall@k across all queries."""
    recalls = []
    for gt, pred in zip(ground_truth, predictions):
        gt_set = set(gt[:k])
        pred_set = set(pred[:k])
        recalls.append(len(gt_set & pred_set) / k)
    return np.mean(recalls)

# Load test data (run generate_test_embeddings.py first)
embeddings = np.load("ch04-vector-storage/embeddings.npy")
queries = np.load("ch04-vector-storage/queries.npy")
num_queries, dimension = queries.shape
k = 10

# Flat (brute-force) -- ground truth
index_flat = faiss.IndexFlatIP(dimension)
index_flat.add(embeddings)
start = time.perf_counter()
distances_flat, indices_flat = index_flat.search(queries, k)
elapsed_flat = (time.perf_counter() - start) * 1000

# IVF
quantizer = faiss.IndexFlatIP(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, 100)
index_ivf.train(embeddings)
index_ivf.add(embeddings)
index_ivf.nprobe = 10
start = time.perf_counter()
distances_ivf, indices_ivf = index_ivf.search(queries, k)
elapsed_ivf = (time.perf_counter() - start) * 1000

# HNSW
index_hnsw = faiss.IndexHNSWFlat(dimension, 32)
index_hnsw.hnsw.efConstruction = 200
index_hnsw.hnsw.efSearch = 64
start_build = time.perf_counter()
index_hnsw.add(embeddings)
build_time = time.perf_counter() - start_build
start = time.perf_counter()
distances_hnsw, indices_hnsw = index_hnsw.search(queries, k)
elapsed_hnsw = (time.perf_counter() - start) * 1000

recall_ivf = recall_at_k(indices_flat, indices_ivf, k=10)
recall_hnsw = recall_at_k(indices_flat, indices_hnsw, k=10)

print(f"\n{'Index':<12} {'Recall@10':<12} {'Latency (ms)':<15} {'Build time'}")
print("-" * 55)
print(f"{'Flat':<12} {'1.000':<12} {elapsed_flat:<15.1f} {'instant'}")
print(f"{'IVF':<12} {recall_ivf:<12.3f} {elapsed_ivf:<15.1f} {'~2-5s'}")
print(f"{'HNSW':<12} {recall_hnsw:<12.3f} {elapsed_hnsw:<15.1f} {f'{build_time:.0f}s'}")
