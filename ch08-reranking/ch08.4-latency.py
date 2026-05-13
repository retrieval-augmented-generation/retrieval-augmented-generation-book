# ch08.4-latency.py - Reranker latency profile as a function of candidate count
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from reranker import rerank

QUERY = "What is the bereavement leave policy?"
TRIALS = 3
CANDIDATE_COUNTS = (5, 10, 20, 50, 100)


def main():
    print(f"Query: {QUERY!r}")
    print(f"Trials per K: {TRIALS}  (median reported)")
    print()
    print(f"{'K':<8}{'pool fetch (ms)':<20}{'rerank (ms)':<16}{'total (ms)':<14}")
    print("-" * 58)

    for k in CANDIDATE_COUNTS:
        fetch_ms, rerank_ms = [], []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            candidates = hybrid_candidates(QUERY, k=k)
            t_fetch = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            rerank(QUERY, candidates, top_n=min(5, k))
            t_rerank = (time.perf_counter() - t0) * 1000

            fetch_ms.append(t_fetch)
            rerank_ms.append(t_rerank)

        fetch_ms.sort()
        rerank_ms.sort()
        med_fetch = fetch_ms[len(fetch_ms) // 2]
        med_rerank = rerank_ms[len(rerank_ms) // 2]
        total = med_fetch + med_rerank
        print(f"{k:<8}{med_fetch:<20.1f}{med_rerank:<16.1f}{total:<14.1f}")


if __name__ == "__main__":
    main()
