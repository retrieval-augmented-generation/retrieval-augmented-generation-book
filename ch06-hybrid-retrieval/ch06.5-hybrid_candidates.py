# ch06.5-hybrid_candidates.py - Build a hybrid candidate pool and inspect it
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import keyword_search, semantic_search, hybrid_candidates

QUERY = "What is the HIPAA Section 164.312(a)(1) requirement?"
K = 20


def main():
    print(f"Query: {QUERY!r}\n")

    kw = keyword_search(QUERY, k=K)
    sem = semantic_search(QUERY, k=K)
    merged = hybrid_candidates(QUERY, k=K)

    both = sum(1 for c in merged if c["matched_by"] == "both")
    kw_only = sum(1 for c in merged if c["matched_by"] == "keyword")
    sem_only = sum(1 for c in merged if c["matched_by"] == "semantic")

    print(f"Keyword candidates:    {len(kw)}")
    print(f"Semantic candidates:   {len(sem)}")
    print(f"Merged candidates:     {len(merged)}")
    print(f"  matched by both:     {both}")
    print(f"  keyword only:        {kw_only}")
    print(f"  semantic only:       {sem_only}")
    print()

    # Sort: matched-by-both first, then by best available rank.
    def best_rank(c: dict) -> int:
        ranks = [r for r in (c.get("keyword_rank"), c.get("semantic_rank")) if r]
        return min(ranks) if ranks else 999

    merged.sort(
        key=lambda c: (0 if c["matched_by"] == "both" else 1, best_rank(c))
    )

    print("Top 10 candidates (matched-by-both first, then best rank):")
    print(f"  {'kw':<5}{'sem':<5}{'tag':<10}source")
    print(f"  {'-'*5}{'-'*5}{'-'*10}{'-'*40}")
    for c in merged[:10]:
        kw_r = c["keyword_rank"] or "-"
        sem_r = c["semantic_rank"] or "-"
        print(
            f"  {str(kw_r):<5}{str(sem_r):<5}{c['matched_by']:<10}"
            f"{c['source_file']} (chunk {c['chunk_index']})"
        )


if __name__ == "__main__":
    main()
