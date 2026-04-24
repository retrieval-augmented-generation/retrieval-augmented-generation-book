# rag_pipeline_test.py -- Compare the same questions through a simple RAG pipeline.
from dotenv import load_dotenv
load_dotenv()
# This is a self-contained demo. You will build each component properly in chapters 2-7.

import json
from pathlib import Path

import numpy as np
from openai import OpenAI

client = OpenAI()

# ---------------------------------------------------------------------------
# Step 0: Load the pre-built index (one embedding per document)
# ---------------------------------------------------------------------------
# The index was built once by scripts/build_index.py so you don't need to
# re-embed the corpus on every run. It covers the full 108-document corpus.
# Chapter 2 introduces chunking and loads the chunk-level index instead.

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"

with open(EMBEDDINGS_DIR / "docs.json", encoding="utf-8") as f:
    documents = json.load(f)

doc_embeddings = np.load(EMBEDDINGS_DIR / "doc_embeddings.npy")

print(f"Loaded {len(documents)} documents from {EMBEDDINGS_DIR}\n")

# ---------------------------------------------------------------------------
# Step 1: Embed a query and retrieve the most relevant chunk
# ---------------------------------------------------------------------------

def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return np.array(response.data[0].embedding)


def retrieve(question: str) -> dict:
    query_emb = embed(question)
    similarities = np.dot(doc_embeddings, query_emb) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    best_idx = int(np.argmax(similarities))
    return documents[best_idx]


# ---------------------------------------------------------------------------
# Step 2: For each question, retrieve the best chunk and ask the LLM
# ---------------------------------------------------------------------------
questions = [
    "How do I get my money back?",
    "What is the SLA uptime guarantee?",
    "How does the deployment pipeline work?",
    "How do I authenticate API requests?",
    "What does CVE-2024-1234 affect in Acme Corp's systems?",
]

for i, question in enumerate(questions, 1):
    doc = retrieve(question)

    prompt = f"""Answer the question based ONLY on the provided context.
If the context does not contain enough information, say "I don't know."
Keep your answer to one or two sentences. Cite the source document in your answer.

Context:
{doc["text"]}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=80,
    )

    print(f"Q{i}: {question}")
    print(f"A{i}: {response.choices[0].message.content}")
    print(f"Source: {doc['source']}")
    print()
