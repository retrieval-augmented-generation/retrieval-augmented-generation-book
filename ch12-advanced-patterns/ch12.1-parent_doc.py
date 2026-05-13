# ch12.1-parent_doc.py - Parent-document retrieval: index small chunks, return larger context
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from database import get_connection
from embedder import embed_query

QUERY = "What is the data retention period for EU customer personal data?"
CHILD_TOP_K = 1
NEIGHBOR_RADIUS = 2  # Chunks before and after the matched child


def semantic_top_chunks(query: str, k: int) -> list[dict]:
    embedding = embed_query(query)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, content, source_file, chunk_index,
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
        {"chunk_id": r[0], "content": r[1], "source_file": r[2],
         "chunk_index": r[3], "score": float(r[4])}
        for r in rows
    ]


def expand_to_parent(child: dict, radius: int) -> list[dict]:
    """Fetch the child chunk plus its neighbors from the same source file."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_index, content
            FROM chunks
            WHERE source_file = %s
              AND chunk_index BETWEEN %s AND %s
            ORDER BY chunk_index
            """,
            (child["source_file"],
             child["chunk_index"] - radius,
             child["chunk_index"] + radius),
        )
        rows = cur.fetchall()
    conn.close()
    return [{"chunk_index": r[0], "content": r[1]} for r in rows]


def main():
    print(f"Query: {QUERY}\n")

    child = semantic_top_chunks(QUERY, k=CHILD_TOP_K)[0]
    print(f"Child chunk (top semantic match):")
    print(f"  {child['source_file']}#chunk-{child['chunk_index']} "
          f"(score {child['score']:.3f})")
    print(f"  Length: {len(child['content'])} chars")
    print(f"  Preview: {child['content'][:200]!r}")
    print()

    parent = expand_to_parent(child, radius=NEIGHBOR_RADIUS)
    parent_text = "\n".join(c["content"] for c in parent)
    parent_indices = [c["chunk_index"] for c in parent]
    print(f"Parent context ({2 * NEIGHBOR_RADIUS + 1} adjacent chunks: {parent_indices}):")
    print(f"  Length: {len(parent_text)} chars  ({len(parent_text) // len(child['content'])}x larger)")
    print(f"  Preview: {parent_text[:300]!r}")
    print()

    print("Child chunk gives the LLM a precise hit. Parent context gives it the")
    print("surrounding paragraph, which often contains the qualifications and")
    print("exceptions that turn a precise hit into a complete answer.")


if __name__ == "__main__":
    main()
