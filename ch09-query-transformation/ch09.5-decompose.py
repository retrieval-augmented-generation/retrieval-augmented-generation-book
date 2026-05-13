# ch09.5-decompose.py - Decompose a multi-part question into sub-questions
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from query_transform import decompose

QUERY = (
    "What are the SLA uptime guarantees, the response time targets, "
    "and how are credits calculated when targets are missed?"
)


def show_sources(candidates: list[dict], n: int = 3) -> None:
    for c in candidates[:n]:
        print(f"    {c['source_file']}#chunk-{c['chunk_index']}")


def main():
    print(f"Compound question:\n  {QUERY}\n")

    print("Single-shot retrieval (top 5 of one hybrid pool):")
    flat = hybrid_candidates(QUERY, k=20)
    show_sources(flat, n=5)
    print()

    subs = decompose(QUERY)
    print(f"Decomposed into {len(subs)} sub-questions:")
    for i, s in enumerate(subs, 1):
        print(f"  {i}. {s}")
    print()

    print("Top-3 retrieval per sub-question:")
    for s in subs:
        print(f"  Sub-question: {s}")
        candidates = hybrid_candidates(s, k=10)
        show_sources(candidates, n=3)
        print()


if __name__ == "__main__":
    main()
