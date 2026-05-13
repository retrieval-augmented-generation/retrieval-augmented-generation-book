# ch06.7-hybrid_pipeline.py - Final demo. Hybrid candidate retrieval with filters.
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import keyword_search, semantic_search, hybrid_candidates


def parse_filter(values: list[str] | None) -> dict | None:
    if not values:
        return None
    return dict(v.split("=", 1) for v in values)


def main():
    parser = argparse.ArgumentParser(description="Hybrid retrieval candidate pool")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=20,
                        help="top-K per retriever before merging")
    parser.add_argument(
        "--filter", action="append",
        help="column=value, repeatable (e.g. --filter doc_type=pdf)"
    )
    args = parser.parse_args()
    filters = parse_filter(args.filter)

    kw = keyword_search(args.query, k=args.k, filters=filters)
    sem = semantic_search(args.query, k=args.k, filters=filters)
    merged = hybrid_candidates(args.query, k=args.k, filters=filters)

    print(f"Query: {args.query!r}")
    if filters:
        print(f"Filters: {filters}")
    print()
    print(f"Keyword candidates:    {len(kw)}")
    print(f"Semantic candidates:   {len(sem)}")
    print(f"Merged candidates:     {len(merged)}")
    counts = {
        tag: sum(1 for c in merged if c["matched_by"] == tag)
        for tag in ("both", "keyword", "semantic")
    }
    print(f"  matched by both:     {counts['both']}")
    print(f"  keyword only:        {counts['keyword']}")
    print(f"  semantic only:       {counts['semantic']}")
    print()

    # Order: matched-by-both first, then by best available rank.
    def best_rank(c: dict) -> int:
        ranks = [r for r in (c.get("keyword_rank"), c.get("semantic_rank")) if r]
        return min(ranks) if ranks else 999

    merged.sort(
        key=lambda c: (0 if c["matched_by"] == "both" else 1, best_rank(c))
    )

    print(
        "Top 5 candidates (pool order, no reranker yet — see Chapter 8):"
    )
    print(f"  {'kw':<5}{'sem':<5}{'tag':<10}source")
    print(f"  {'-'*5}{'-'*5}{'-'*10}{'-'*40}")
    for c in merged[:5]:
        kw_r = c["keyword_rank"] or "-"
        sem_r = c["semantic_rank"] or "-"
        print(
            f"  {str(kw_r):<5}{str(sem_r):<5}{c['matched_by']:<10}"
            f"{c['source_file']} (chunk {c['chunk_index']})"
        )


if __name__ == "__main__":
    main()
