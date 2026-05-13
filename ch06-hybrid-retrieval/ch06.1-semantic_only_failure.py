# ch06.1-semantic_only_failure.py - Semantic search misses an exact identifier
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import semantic_search

QUERY = "What is the HIPAA Section 164.312(a)(1) requirement?"
CANONICAL_NEEDLE = "164.312(a)(1)"


def main():
    print(f"Query: {QUERY}\n")

    top = semantic_search(QUERY, k=10)
    print("Semantic top 10:")
    print(f"  {'rank':<5}{'score':<10}source")
    print(f"  {'-'*5}{'-'*10}{'-'*40}")
    for r in top:
        print(
            f"  {r['semantic_rank']:<5}{r['semantic_score']:<10.4f}"
            f"{r['source_file']} (chunk {r['chunk_index']})"
        )

    # Find where the canonical chunk actually ranks by scanning further.
    extended = semantic_search(QUERY, k=30)
    matches = [
        r for r in extended if CANONICAL_NEEDLE in r["content"]
    ]
    print()
    if not matches:
        print(f"Canonical chunk (contains '{CANONICAL_NEEDLE}'): not in top 30")
        return

    first = min(matches, key=lambda r: r["semantic_rank"])
    print(
        f"Canonical chunk containing '{CANONICAL_NEEDLE}': "
        f"{first['source_file']} (chunk {first['chunk_index']})"
    )
    print(f"  semantic rank: {first['semantic_rank']}")
    print(f"  semantic score: {first['semantic_score']:.4f}")
    in_top_10 = first["semantic_rank"] <= 10
    print(f"  in top 10? {'yes' if in_top_10 else 'no'}")


if __name__ == "__main__":
    main()
