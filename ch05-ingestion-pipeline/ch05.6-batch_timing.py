# batch_timing.py -- Compare single vs batch embedding speed
from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = ["This is a sample chunk of text for embedding."] * 500

# One at a time
start = time.time()
for text in texts:
    model.encode([text])
single_time = time.time() - start

# Batched
start = time.time()
model.encode(texts, batch_size=64)
batch_time = time.time() - start

print(f"Single embedding:  {single_time:.1f}s")
print(f"Batch embedding:   {batch_time:.1f}s")
print(f"Speedup:           {single_time / batch_time:.1f}x")
