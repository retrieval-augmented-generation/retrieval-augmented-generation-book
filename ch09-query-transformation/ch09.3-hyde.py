# ch09.3-hyde.py - Hypothetical Document Embeddings for short or abstract queries
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from database import get_connection
from embedder import embed_query
from hybrid import hybrid_candidates
from query_transform import hyde

QUERY = "on-call"


def semantic_search_for_text(text: str, k: int = 20) -> list[dict]:
    """Semantic search using the embedding of arbitrary text (not the original query)."""
    embedding = embed_query(text)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, content, source_file, doc_type, chunk_index,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, embedding, k),
        )
        rows = cur.fetchall()
    conn.close()
    return [
        {
            "chunk_id": r[0], "content": r[1], "source_file": r[2],
            "doc_type": r[3], "chunk_index": r[4],
            "semantic_rank": i + 1, "semantic_score": float(r[5]),
        }
        for i, r in enumerate(rows)
    ]


def show_top(label: str, candidates: list[dict], n: int = 5) -> None:
    print(f"{label}:")
    for i, c in enumerate(candidates[:n], 1):
        print(
            f"  {i}. {c['source_file']}#chunk-{c['chunk_index']}"
        )
    print()


def main():
    print(f"Query: {QUERY!r}\n")

    orig = hybrid_candidates(QUERY, k=20)
    show_top("Original hybrid top 5", orig)

    passage = hyde(QUERY)
    print(f"HyDE passage:")
    print(f"  {passage}")
    print()

    hyde_results = semantic_search_for_text(passage, k=20)
    show_top("HyDE semantic top 5 (using passage embedding)", hyde_results)

    orig_sources = {c["source_file"] for c in orig[:5]}
    hyde_sources = {c["source_file"] for c in hyde_results[:5]}
    print(f"Sources only in original top 5: {sorted(orig_sources - hyde_sources)}")
    print(f"Sources only in HyDE top 5: {sorted(hyde_sources - orig_sources)}")


if __name__ == "__main__":
    main()
