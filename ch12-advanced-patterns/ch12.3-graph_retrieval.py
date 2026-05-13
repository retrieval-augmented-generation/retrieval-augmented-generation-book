# ch12.3-graph_retrieval.py - Graph traversal as a retrieval strategy for entity-rich queries.
# The graph here is hand-authored. In production it would come from cross-references,
# structured metadata, or extraction; the retrieval pattern is the same either way.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from database import get_connection
from embedder import embed_query


# Hand-authored teaching graph. Nodes are real Acme Corp documents; edges are
# relationships an internal-knowledge graph might contain. This is small on
# purpose. A production graph would be built from explicit cross-references,
# structured metadata, or evaluated entity extraction.
EDGES: dict[str, list[tuple[str, str]]] = {
    "customer_deletion_workflow.md": [
        ("references",     "data_retention_policy.md"),
        ("constrained_by", "regulatory_gdpr_summary.pdf"),
        ("audited_by",     "audit_log_retention.md"),
    ],
    "data_retention_policy.md": [
        ("constrained_by", "regulatory_gdpr_summary.pdf"),
        ("exception_for",  "hr_data_retention_schedule.md"),
    ],
    "privacy_policy.html": [
        ("constrained_by", "regulatory_gdpr_summary.pdf"),
    ],
}


def traverse(seed: str, depth: int) -> list[tuple[str, str, str]]:
    """Breadth-first traversal up to depth. Returns (from, edge, to) triples."""
    visited: set[str] = {seed}
    frontier = [seed]
    triples: list[tuple[str, str, str]] = []
    for _ in range(depth):
        next_frontier: list[str] = []
        for node in frontier:
            for edge, target in EDGES.get(node, []):
                triples.append((node, edge, target))
                if target not in visited:
                    visited.add(target)
                    next_frontier.append(target)
        frontier = next_frontier
    return triples


def chunks_from(source_files: list[str], k_per_source: int) -> list[dict]:
    """Fetch the first k_per_source chunks per source file. Order: chunk_index."""
    conn = get_connection()
    rows: list = []
    with conn.cursor() as cur:
        for src in source_files:
            cur.execute(
                """
                SELECT source_file, chunk_index, content
                FROM chunks
                WHERE source_file = %s
                ORDER BY chunk_index
                LIMIT %s
                """,
                (src, k_per_source),
            )
            rows.extend(cur.fetchall())
    conn.close()
    return [{"source_file": r[0], "chunk_index": r[1], "content": r[2]} for r in rows]


def hybrid_top_sources(query: str, k: int = 5) -> list[str]:
    """Top sources from a normal hybrid retrieval, for comparison."""
    embedding = embed_query(query)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source_file
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, k * 3),
        )
        rows = cur.fetchall()
    conn.close()
    seen: list[str] = []
    for (src,) in rows:
        if src not in seen:
            seen.append(src)
        if len(seen) >= k:
            break
    return seen


QUERY = "How does customer deletion interact with retention and GDPR requirements?"
SEED = "customer_deletion_workflow.md"
DEPTH = 2


def main():
    print(f"Query: {QUERY!r}\n")

    print(f"Seed document (from hybrid retrieval or external selection): {SEED}\n")

    triples = traverse(SEED, depth=DEPTH)
    print(f"Graph traversal depth={DEPTH}:")
    print(f"  {SEED}")
    for src, edge, dst in triples:
        if src == SEED:
            print(f"    -> {dst:<40} ({edge})")
        else:
            print(f"    -> {dst:<40} (via {src}, {edge})")
    print()

    connected = sorted({SEED} | {dst for _, _, dst in triples})
    print(f"Evidence set ({len(connected)} connected documents):")
    for src in connected:
        print(f"  {src}")
    print()

    candidate_chunks = chunks_from(connected, k_per_source=2)
    print(f"Chunks aggregated from connected documents: {len(candidate_chunks)}")
    print()

    print("Comparison with normal hybrid retrieval top sources:")
    baseline_sources = hybrid_top_sources(QUERY, k=5)
    for s in baseline_sources:
        marker = "*" if s in connected else " "
        print(f"  {marker} {s}")
    print()
    print("'*' marks sources the graph traversal also selected.")
    print("Graph traversal pulls in connected documents by relationship,")
    print("not by similarity. That is its strength on entity-relationship")
    print("queries and its weakness on queries where the graph is sparse")
    print("or noisy.")


if __name__ == "__main__":
    main()
