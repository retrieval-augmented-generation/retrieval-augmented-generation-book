# ch06.3-keyword_vs_semantic.py - Side-by-side: each retriever wins different queries
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import keyword_search, semantic_search


QUERIES = [
    ("Exact identifier",
     "What is the HIPAA Section 164.312(a)(1) requirement?"),
    ("Paraphrase",
     "How long do we keep customer information?"),
]


def print_results(label: str, results: list[dict]) -> None:
    if not results:
        print(f"  {label}: no results")
        return
    print(f"  {label}:")
    for r in results[:5]:
        rank = r.get("keyword_rank") or r.get("semantic_rank")
        score = r.get("keyword_score")
        if score is None:
            score = r.get("semantic_score")
        print(
            f"    {rank}. [{score:.4f}] {r['source_file']} "
            f"(chunk {r['chunk_index']})"
        )


def main():
    for kind, query in QUERIES:
        print("=" * 60)
        print(f"{kind} query: {query!r}")
        print("=" * 60)

        kw = keyword_search(query, k=5)
        sem = semantic_search(query, k=5)

        print_results("Keyword (FTS, ts_rank_cd)", kw)
        print()
        print_results("Semantic (pgvector cosine)", sem)
        print()


if __name__ == "__main__":
    main()
