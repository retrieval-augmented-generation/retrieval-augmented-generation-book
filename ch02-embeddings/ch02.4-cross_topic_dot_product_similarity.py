import sys
sys.path.insert(0, "ch02-embeddings")
import numpy as np
from sentence_transformers import SentenceTransformer
from corpus import documents

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

# Within-topic: two Operations sentences (SLA uptime vs refund policy)
print(f"Ops vs Ops:           {np.dot(embeddings[0], embeddings[1]):.4f}")
# Within-topic: two IT & Security sentences (audit logs vs backups)
print(f"IT vs IT:             {np.dot(embeddings[12], embeddings[13]):.4f}")
# Cross-topic: Operations vs IT & Security
print(f"Ops vs IT:            {np.dot(embeddings[0], embeddings[12]):.4f}")
# Cross-topic: Operations vs Compliance
print(f"Ops vs Compliance:    {np.dot(embeddings[0], embeddings[20]):.4f}")
