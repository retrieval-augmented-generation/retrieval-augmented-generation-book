# pgvector_setup.py
import psycopg2
import numpy as np
import time

# Load test data (run generate_test_embeddings.py first)
embeddings = np.load("ch04-vector-storage/embeddings.npy")
queries = np.load("ch04-vector-storage/queries.npy")
dimension = embeddings.shape[1]

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="rag",
    user="rag",
    password="rag"
)
conn.autocommit = True
cur = conn.cursor()

# Enable the pgvector extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create a table with a vector column
cur.execute(f"""
    DROP TABLE IF EXISTS embeddings;
    CREATE TABLE embeddings (
        id SERIAL PRIMARY KEY,
        embedding vector({dimension})
    );
""")

print("Table created. Inserting embeddings...")

# Insert embeddings in batches
batch_size = 1000
start_insert = time.perf_counter()
for i in range(0, len(embeddings), batch_size):
    batch = embeddings[i:i + batch_size]
    values = [(vec.tolist(),) for vec in batch]
    args_str = ",".join(
        cur.mogrify("(%s)", (vec,)).decode() for vec in [v[0] for v in values]
    )
    cur.execute(f"INSERT INTO embeddings (embedding) VALUES {args_str}")

insert_time = time.perf_counter() - start_insert
print(f"Inserted {len(embeddings)} vectors in {insert_time:.1f}s")
