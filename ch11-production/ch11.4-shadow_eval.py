# ch11.4-shadow_eval.py - Run two pipeline variants on the same queries, compare metrics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from rag import answer

QUERIES = [
    "What is the HIPAA Section 164.312(a)(1) requirement?",
    "What is the bereavement leave policy?",
    "What is the enterprise SLA uptime guarantee?",
]


def run_variant(label: str, use_reranker: bool, use_query_transform: bool) -> list[dict]:
    config.USE_RERANKER = use_reranker
    config.USE_QUERY_TRANSFORM = use_query_transform
    results = []
    for q in QUERIES:
        result = answer(q)
        stages = result["trace"]["stages"]
        cite_acc = stages.get("generation.complete", {}).get("citation_accuracy", 1.0)
        results.append({
            "query": q,
            "total_latency_ms": result["trace"]["total_latency_ms"],
            "cite_acc": cite_acc,
        })
    return results


def main():
    print("Shadow comparison: run the same queries through two pipeline variants.\n")

    current = run_variant("Current", use_reranker=False, use_query_transform=False)
    shadow = run_variant("Shadow",  use_reranker=True,  use_query_transform=False)

    print(f"{'Query':<60}{'Cur ms':<10}{'Sha ms':<10}{'Cur cite':<10}{'Sha cite':<10}")
    print("-" * 100)
    for c, s in zip(current, shadow):
        print(
            f"{c['query'][:58]:<60}"
            f"{c['total_latency_ms']:<10.0f}{s['total_latency_ms']:<10.0f}"
            f"{c['cite_acc']:<10.2f}{s['cite_acc']:<10.2f}"
        )

    print()
    print("Current pipeline is what serves users today; shadow runs in parallel")
    print("on the same query stream and emits its own traces. Compare metrics")
    print("before promoting shadow to current.")


if __name__ == "__main__":
    main()
