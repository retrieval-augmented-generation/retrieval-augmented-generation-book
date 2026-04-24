# search.py -- Brute-force similarity search
from sentence_transformers import SentenceTransformer
import numpy as np
from corpus import documents

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

def search(query: str, top_k: int = 5) -> list[tuple[int, float, str]]:
    """Embed a query and return the top-K most similar documents."""
    query_embedding = model.encode([query])[0]
    # For normalized vectors, cosine similarity = dot product
    similarities = np.dot(embeddings, query_embedding)
    # Get indices of top-K highest similarities
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append((idx, float(similarities[idx]), documents[idx]))
    return results

def print_results(query: str, results: list[tuple[int, float, str]]):
    """Display search results with scores."""
    print(f"\nQuery: {query}")
    print("-" * 80)
    for rank, (idx, score, doc) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}] {doc}")
    print()
