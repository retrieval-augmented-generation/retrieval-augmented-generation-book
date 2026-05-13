# ch12.2-contextual_chunks.py - Contextual retrieval: prepend an LLM-generated context
# prefix to each chunk before embedding. Demo limits scope to one document so the
# index-time cost stays small.
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

from database import get_connection
from embedder import embed_query, model as embed_model

load_dotenv()
client = OpenAI()
LLM_MODEL = "gpt-4o-mini"

# One document, narrowly scoped. Contextual retrieval at full-corpus scale is an
# indexing project. The demo shows what one document's worth of contextual
# embedding looks like.
SOURCE_DOCUMENT = "customer_deletion_workflow.md"
QUERY = "When can deleted customer records be retained for legal reasons?"

CONTEXT_PROMPT = """You are preparing a chunk of a corporate document for retrieval. Write a brief one-sentence context prefix that explains where this chunk sits inside the parent document and what it is about. The prefix will be prepended to the chunk's text before the chunk is embedded for semantic search.

Document name: {document}

Chunk text:
{chunk}

One-sentence context prefix:"""


def fetch_document_chunks(source: str) -> list[dict]:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, content, source_file, chunk_index, embedding
            FROM chunks
            WHERE source_file = %s
            ORDER BY chunk_index
            """,
            (source,),
        )
        rows = cur.fetchall()
    conn.close()
    return [
        {"chunk_id": r[0], "content": r[1], "source_file": r[2],
         "chunk_index": r[3], "embedding": np.array(r[4])}
        for r in rows
    ]


def generate_prefix(chunk: dict) -> str:
    prompt = CONTEXT_PROMPT.format(document=chunk["source_file"], chunk=chunk["content"][:1500])
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def top_k_cosine(query_emb: np.ndarray, chunks: list[dict], emb_key: str, k: int) -> list[dict]:
    """Brute-force cosine ranking over a small chunk list."""
    scored = []
    q = query_emb / np.linalg.norm(query_emb)
    for c in chunks:
        v = c[emb_key]
        v = v / np.linalg.norm(v)
        scored.append((float(np.dot(q, v)), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [dict(c, score=s) for s, c in scored[:k]]


def main():
    print(f"Source document: {SOURCE_DOCUMENT}")
    print(f"Query: {QUERY!r}\n")

    chunks = fetch_document_chunks(SOURCE_DOCUMENT)
    print(f"Fetched {len(chunks)} chunks from the source document.")
    print()

    # Generate contextual prefixes for every chunk
    print("Generating contextual prefixes (one LLM call per chunk)...")
    for c in chunks:
        prefix = generate_prefix(c)
        c["prefix"] = prefix
        c["contextual_text"] = prefix + "\n\n" + c["content"]

    # Show two prefixes so the reader sees what the LLM produced
    sample_indices = [1, min(8, len(chunks) - 1)]
    for i in sample_indices:
        print(f"\n--- Example: chunk {chunks[i]['chunk_index']} ---")
        print(f"Original (first 160 chars):")
        print(f"  {chunks[i]['content'][:160]!r}")
        print(f"Generated context prefix:")
        print(f"  {chunks[i]['prefix']!r}")
    print()

    # Re-embed the contextualized text
    print("Re-embedding contextualized chunks locally...")
    contextual_embeddings = embed_model.encode(
        [c["contextual_text"] for c in chunks], show_progress_bar=False
    )
    for c, e in zip(chunks, contextual_embeddings):
        c["contextual_embedding"] = e
    print()

    # Compare retrieval
    query_emb = np.array(embed_query(QUERY))

    baseline = top_k_cosine(query_emb, chunks, "embedding", k=5)
    contextual = top_k_cosine(query_emb, chunks, "contextual_embedding", k=5)

    print("Baseline retrieval (original embeddings, top 5):")
    for c in baseline:
        print(f"  [{c['score']:.4f}] chunk-{c['chunk_index']:<3} "
              f"{c['content'][:80]!r}")
    print()
    print("Contextual retrieval (re-embedded with LLM prefix, top 5):")
    for c in contextual:
        print(f"  [{c['score']:.4f}] chunk-{c['chunk_index']:<3} "
              f"{c['content'][:80]!r}")
    print()

    baseline_ids = [c["chunk_index"] for c in baseline]
    contextual_ids = [c["chunk_index"] for c in contextual]
    moved_in = [i for i in contextual_ids if i not in baseline_ids]
    moved_out = [i for i in baseline_ids if i not in contextual_ids]
    print(f"Chunks promoted into top 5 by contextualization: {moved_in}")
    print(f"Chunks demoted out of top 5 by contextualization: {moved_out}")


if __name__ == "__main__":
    main()
