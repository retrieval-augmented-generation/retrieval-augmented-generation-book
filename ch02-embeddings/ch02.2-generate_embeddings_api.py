from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
import numpy as np
from corpus import documents

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents
)

embeddings = np.array([item.embedding for item in response.data])

print(f"Type: {type(embeddings)}")
print(f"Shape: {embeddings.shape}")
print(f"Dtype: {embeddings.dtype}")
print(f"First embedding (first 10 values): {embeddings[0][:10]}")
