# ch09.1-bad_query.py - Three query shapes that retrieval handles unevenly
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates

# Three query shapes that motivate the rest of the chapter.
QUERIES = [
    ("Sparse / single-term",
     "MFA"),
    ("Colloquial phrasing",
     "I gotta send some sensitive data to a partner, can I just email it?"),
    ("Compound / multi-part",
     "What are the SLA uptime guarantees, the response time targets, "
     "and how are credits calculated when targets are missed?"),
]


def top_sources(candidates: list[dict], n: int = 5) -> list[str]:
    return [
        f"{c['source_file']}#chunk-{c['chunk_index']}" for c in candidates[:n]
    ]


def main():
    for shape, query in QUERIES:
        print("=" * 70)
        print(f"{shape}")
        print(f"Query: {query!r}")
        print("=" * 70)
        candidates = hybrid_candidates(query, k=20)
        print(f"Candidate pool: {len(candidates)}")
        print("Hybrid top 5:")
        for src in top_sources(candidates, n=5):
            print(f"  {src}")
        print()


if __name__ == "__main__":
    main()
