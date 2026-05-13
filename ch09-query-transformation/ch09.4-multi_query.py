# ch09.4-multi_query.py - Multi-query expansion for ambiguous queries
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from query_transform import multi_query

QUERY = "phishing"
N_VARIATIONS = 3


def main():
    print(f"Original query: {QUERY!r}\n")

    orig = hybrid_candidates(QUERY, k=10)
    print("Original hybrid top 5:")
    for i, c in enumerate(orig[:5], 1):
        print(f"  {i}. {c['source_file']}#chunk-{c['chunk_index']}")
    print()
    orig_ids = {c["chunk_id"] for c in orig}

    variations = multi_query(QUERY, n=N_VARIATIONS)
    print(f"{N_VARIATIONS} multi-query variations:")
    for v in variations:
        print(f"  - {v}")
    print()

    # Union pool across original + variations
    seen: set[int] = set(orig_ids)
    union = list(orig)
    new_per_variation: list[tuple[str, int]] = []
    for v in variations:
        added = 0
        for c in hybrid_candidates(v, k=10):
            if c["chunk_id"] not in seen:
                union.append(c)
                seen.add(c["chunk_id"])
                added += 1
        new_per_variation.append((v, added))

    print("New chunks added per variation:")
    for v, count in new_per_variation:
        print(f"  +{count:>2}  {v}")
    print()
    print(f"Original pool size:        {len(orig_ids)}")
    print(f"Union pool size:           {len(seen)}")
    print(f"Net new chunks from expansion: {len(seen) - len(orig_ids)}")


if __name__ == "__main__":
    main()
