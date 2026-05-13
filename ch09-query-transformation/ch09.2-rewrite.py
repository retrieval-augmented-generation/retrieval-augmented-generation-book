# ch09.2-rewrite.py - Rewrite a colloquial query into formal document-style language
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from query_transform import rewrite

QUERY = "I gotta send some sensitive data to a partner, can I just email it?"


def show_top(label: str, candidates: list[dict], n: int = 5) -> None:
    print(f"{label}:")
    for i, c in enumerate(candidates[:n], 1):
        kw = c.get("keyword_rank") or "-"
        sem = c.get("semantic_rank") or "-"
        print(
            f"  {i}. kw={kw}/sem={sem}  "
            f"{c['source_file']}#chunk-{c['chunk_index']}"
        )
    print()


def main():
    print(f"Original query: {QUERY!r}\n")
    orig = hybrid_candidates(QUERY, k=20)
    show_top("Original hybrid top 5", orig)

    rewritten = rewrite(QUERY)
    print(f"Rewritten query: {rewritten!r}\n")
    rw = hybrid_candidates(rewritten, k=20)
    show_top("Rewritten hybrid top 5", rw)

    orig_sources = {c["source_file"] for c in orig[:5]}
    rw_sources = {c["source_file"] for c in rw[:5]}
    print(f"Sources only in original top 5: {sorted(orig_sources - rw_sources)}")
    print(f"Sources only in rewritten top 5: {sorted(rw_sources - orig_sources)}")


if __name__ == "__main__":
    main()
