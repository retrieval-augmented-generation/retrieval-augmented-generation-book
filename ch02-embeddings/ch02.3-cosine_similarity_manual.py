# cosine_similarity.py -- Computing similarity metrics manually
import numpy as np
from sentence_transformers import SentenceTransformer
from corpus import documents

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

# Pick two documents about Operations
doc_a = embeddings[0]  # "Acme Corp guarantees 99.95% uptime..."
doc_b = embeddings[1]  # "Enterprise customers may request a full refund..."

# Cosine similarity from scratch
dot_product = np.dot(doc_a, doc_b)
norm_a = np.linalg.norm(doc_a)
norm_b = np.linalg.norm(doc_b)
cosine_sim = dot_product / (norm_a * norm_b)

print(f"Dot product:        {dot_product:.4f}")
print(f"Norm of A:          {norm_a:.4f}")
print(f"Norm of B:          {norm_b:.4f}")
print(f"Cosine similarity:  {cosine_sim:.4f}")
