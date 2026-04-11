# rag_pipeline_test.py -- Compare the same questions through a simple RAG pipeline.
from dotenv import load_dotenv
load_dotenv()
# This is a self-contained demo. You will build each component properly in chapters 2-7.

import os
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Step 0: Load documents from the Acme Corp corpus
# ---------------------------------------------------------------------------
# We load only the 5 documents needed for this demo. The full corpus (110
# documents) gets ingested properly in chapter 5.

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

# The 5 documents that answer our 5 test questions.
FILENAMES = [
    "enterprise_refund_policy.pdf",
    "pricing_update_october_2024.pdf",
    "enterprise_sla.pdf",
    "q3_internal_memo.md",
    "security_certifications_summary.md",
]


def load_document(path: Path) -> str:
    """Read a document and return its text content."""
    if path.suffix == ".pdf":
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    return path.read_text()


documents = []
for filename in FILENAMES:
    text = load_document(CORPUS_DIR / filename)
    documents.append({"filename": filename, "content": text})

print(f"Loaded {len(documents)} documents from {CORPUS_DIR}\n")

# ---------------------------------------------------------------------------
# Step 1: Embed the documents (this is the "index" step)
# ---------------------------------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_texts = [d["content"] for d in documents]
doc_embeddings = model.encode(doc_texts)

# ---------------------------------------------------------------------------
# Step 2: For each question, retrieve the most relevant document and ask the LLM
# ---------------------------------------------------------------------------
questions = [
    "What is Acme Corp's refund policy for enterprise customers?",
    "When did Acme Corp change its pricing for the Pro tier?",
    "What are the SLA guarantees for Acme Corp's API uptime?",
    "Who approved the Q3 budget reallocation for the ML platform team?",
    "What security certifications does Acme Corp currently hold?",
]

client = OpenAI()

for i, question in enumerate(questions, 1):
    # Retrieve: embed the question and find the closest document
    query_emb = model.encode(question)
    similarities = np.dot(doc_embeddings, query_emb) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    best_idx = np.argmax(similarities)
    source = documents[best_idx]

    # Augment + Generate: build a grounded prompt and call the LLM
    prompt = f"""Answer the question based ONLY on the provided context.
If the context does not contain enough information, say "I don't know."
Cite the source document in your answer.

Context:
{source["content"]}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    print(f"Q{i}: {question}")
    print(f"A{i}: {response.choices[0].message.content}")
    print(f"Source: {source['filename']}")
    print()
