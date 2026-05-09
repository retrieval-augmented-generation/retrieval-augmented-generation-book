# pgvector_no_index.py
# Search WITHOUT an index -- sequential scan (brute-force)
import psycopg2
import numpy as np
import time

queries = np.load("ch04-vector-storage/queries.npy")

conn = psycopg2.connect(host="localhost", port=5432, dbname="rag", user="rag", password="rag")
conn.autocommit = True
cur = conn.cursor()

query_vec = queries[0].tolist()

start = time.perf_counter()
cur.execute("""
    SELECT id, embedding <#> %s::vector AS distance
    FROM embeddings
    ORDER BY distance
    LIMIT 10;
""", (query_vec,))
results_no_index = cur.fetchall()
no_index_time = (time.perf_counter() - start) * 1000

print(f"Sequential scan: {no_index_time:.1f} ms per query")

# Verify it is doing a sequential scan
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
