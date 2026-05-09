# benchmark_pgvector.py
import psycopg2
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

# Connect and ensure index exists
conn = psycopg2.connect(host="localhost", port=5432, dbname="rag", user="rag", password="rag")
conn.autocommit = True
cur = conn.cursor()
cur.execute("SET hnsw.ef_search = 64;")

# Run all test queries
latencies = []
all_pgvector_ids = []

for q in queries:
    query_vec = q.tolist()
    start = time.perf_counter()
    cur.execute("""
        SELECT id
        FROM embeddings
        ORDER BY embedding <#> %s::vector
        LIMIT 10;
    """, (query_vec,))
    results = cur.fetchall()
    latencies.append((time.perf_counter() - start) * 1000)
    # pgvector IDs are 1-indexed, FAISS IDs are 0-indexed
    all_pgvector_ids.append([r[0] - 1 for r in results])

avg_latency = np.mean(latencies)
p99_latency = np.percentile(latencies, 99)
recall_pg = recall_at_k(indices_flat, np.array(all_pgvector_ids), k=10)

print(f"pgvector HNSW: recall@10={recall_pg:.3f}, "
      f"avg={avg_latency:.1f}ms, p99={p99_latency:.1f}ms")

conn.close()
