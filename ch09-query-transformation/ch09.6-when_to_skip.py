# ch09.6-when_to_skip.py - Transformation can hurt when the query is already specific
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from query_transform import rewrite

# Queries that are already specific and well-formed. Transformation here is a
# round-trip through the LLM that adds latency without changing retrieval.
QUERIES = [
    "What is the HIPAA Section 164.312(a)(1) requirement?",
    "What is the password rotation policy?",
    "What is the enterprise SLA uptime guarantee?",
]


def top_sources(candidates: list[dict], n: int = 3) -> list[str]:
    return [f"{c['source_file']}#chunk-{c['chunk_index']}" for c in candidates[:n]]


def main():
    print(f"{'Query':<55}{'Same top 3?':<14}{'rewrite cost':<14}")
    print("-" * 83)

    for q in QUERIES:
        t0 = time.perf_counter()
        rw = rewrite(q)
        rewrite_ms = (time.perf_counter() - t0) * 1000

        orig_top = top_sources(hybrid_candidates(q, k=10), n=3)
        rw_top = top_sources(hybrid_candidates(rw, k=10), n=3)

        same = "yes" if orig_top == rw_top else "no"
        print(f"{q[:53]:<55}{same:<14}{rewrite_ms:.0f} ms")

    print()
    print("Rewriting changed the top 3 for all of these specific queries")
    print("and added 700-900 ms of LLM latency per query. Without a labeled")
    print("evaluation set, those changes are not evidence of improvement.")


if __name__ == "__main__":
    main()
