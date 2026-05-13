# ch08.3-tune_k_n.py - Sweep candidate K and final N, measure recall and latency.
#
# Scope is intentionally small so the demo finishes in a few minutes on CPU.
# Readers with a GPU can scale K (e.g. up to 50 or 100) and the eval set; the
# trend the script demonstrates is the same.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from reranker import rerank


# Small labeled query set. Each tuple is (query, source_file_that_should_appear_in_top_N).
EVAL_SET = [
    ("What is the HIPAA Section 164.312(a)(1) requirement?",
     "hipaa_compliance_overview.md"),
    ("What is the password rotation policy?",
     "it_security_policy.pdf"),
    ("What happens to my unused vacation when I leave the company?",
     "vacation_policy.md"),
    ("What is the bereavement leave policy?",
     "bereavement_leave_policy.md"),
    ("How do I enroll in multi-factor authentication?",
     "mfa_enrollment_guide.html"),
]

K_VALUES = (5, 10, 20)
N_VALUES = (3, 5)


def evaluate_K(k: int) -> dict[int, tuple[float, float]]:
    """For a single K, rerank once per query and slice the result for each N.
    Returns {n: (recall_at_n, avg_latency_seconds_per_query)}."""
    counts = {n: [0, 0.0] for n in N_VALUES if n <= k}
    for query, canonical_source in EVAL_SET:
        t0 = time.perf_counter()
        candidates = hybrid_candidates(query, k=k)
        max_n = min(max(N_VALUES), len(candidates))
        reranked = rerank(query, candidates, top_n=max_n)
        elapsed = time.perf_counter() - t0
        for n in counts:
            top = reranked[:n]
            if any(c["source_file"] == canonical_source for c in top):
                counts[n][0] += 1
            counts[n][1] += elapsed
    return {n: (h / len(EVAL_SET), t / len(EVAL_SET)) for n, (h, t) in counts.items()}


def main():
    print(
        f"Eval set: {len(EVAL_SET)} queries  "
        f"(canonical source must appear in top N)",
        flush=True,
    )
    print(f"{'K':<6}{'N':<6}{'recall@N':<12}{'avg latency (s)':<18}", flush=True)
    print("-" * 42, flush=True)

    for k in K_VALUES:
        per_n = evaluate_K(k)
        for n in N_VALUES:
            if n > k:
                continue
            recall, latency = per_n[n]
            print(
                f"{k:<6}{n:<6}{recall:<12.3f}{latency:<18.3f}", flush=True
            )


if __name__ == "__main__":
    main()
