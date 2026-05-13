# ch08.1-rank_problem.py - The right chunks are in the pool but not in the top 5
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from rag import _interleave

QUERY = "What happens to my unused vacation when I leave the company?"
NEEDLE = "unused vacation"


def main():
    candidates = hybrid_candidates(QUERY, k=20)
    interleaved = _interleave(candidates, n=20)

    canonical = [
        (i + 1, c) for i, c in enumerate(interleaved)
        if NEEDLE.lower() in c["content"].lower()
    ]

    print(f"Query: {QUERY!r}\n")
    print(f"Candidate pool: {len(candidates)} chunks")
    print(f"Chunks containing {NEEDLE!r}: {len(canonical)}\n")

    print("Interleave selection (full pool order):")
    print(f"  {'pos':<5}{'tag':<10}source")
    print(f"  {'-'*5}{'-'*10}{'-'*50}")
    for pos, c in canonical:
        marker = "**" if pos > 5 else "  "
        print(
            f"{marker}{pos:<5}{c['matched_by']:<10}"
            f"{c['source_file']}#chunk-{c['chunk_index']}"
        )

    in_top5 = sum(1 for pos, _ in canonical if pos <= 5)
    in_pool = len(canonical)
    print()
    print(f"Canonical chunks in pool:   {in_pool}")
    print(f"Canonical chunks in top 5:  {in_top5}")
    print(f"Canonical chunks lost to selection: {in_pool - in_top5}")


if __name__ == "__main__":
    main()
