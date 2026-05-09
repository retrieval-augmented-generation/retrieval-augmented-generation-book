# pgvector_create_index.py
# Create an HNSW index and search with it
import psycopg2
import numpy as np
import time

queries = np.load("ch04-vector-storage/queries.npy")

conn = psycopg2.connect(host="localhost", port=5432, dbname="rag", user="rag", password="rag")
conn.autocommit = True
cur = conn.cursor()

print("Creating HNSW index (this takes a few minutes)...")

start_index = time.perf_counter()
cur.execute("""
    CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx ON embeddings
    USING hnsw (embedding vector_ip_ops)
    WITH (m = 32, ef_construction = 200);
""")
index_time = time.perf_counter() - start_index
print(f"Index created in {index_time:.1f}s")

# Set ef_search for queries
cur.execute("SET hnsw.ef_search = 64;")

# Search WITH the index
query_vec = queries[0].tolist()
start = time.perf_counter()
cur.execute("""
    SELECT id, embedding <#> %s::vector AS distance
    FROM embeddings
    ORDER BY distance
    LIMIT 10;
""", (query_vec,))
results_with_index = cur.fetchall()
index_time_query = (time.perf_counter() - start) * 1000

print(f"HNSW indexed search: {index_time_query:.1f} ms per query")

# Verify index usage
cur.execute("""
    EXPLAIN ANALYZE
    SELECT id, embedding <#> %s::vector AS distance
    FROM embeddings
    ORDER BY distance
    LIMIT 10;
""", (query_vec,))
plan = cur.fetchall()
for row in plan:
    print(row[0])

conn.close()
