from sentence_transformers import SentenceTransformer
import numpy as np
from corpus import documents

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

print(f"Type: {type(embeddings)}")
print(f"Shape: {embeddings.shape}")
print(f"Dtype: {embeddings.dtype}")
print(f"First embedding (first 10 values): {embeddings[0][:10]}")
