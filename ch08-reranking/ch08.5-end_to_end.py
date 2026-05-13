# ch08.5-end_to_end.py - Five queries through both pipelines: interleave vs reranker
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag import answer

QUERIES = [
    "What multi-factor authentication methods does Acme Corp support?",
    "When does open enrollment start and what are the plan options?",
    "What is Acme Corp's data retention policy for customer records?",
    "What is the bereavement leave policy?",
    "What happens to my unused vacation when I leave the company?",
]


def main():
    for i, q in enumerate(QUERIES, start=1):
        print("=" * 70)
        print(f"Q{i}: {q}")
        print("=" * 70)

        baseline = answer(q, k=20, n=5, use_reranker=False)
        reranked = answer(q, k=20, n=5, use_reranker=True)

        baseline_chunks = [
            f"{c['source_file']}#chunk-{c['chunk_index']}" for c in baseline["context"]
        ]
        reranked_chunks = [
            f"{c['source_file']}#chunk-{c['chunk_index']}" for c in reranked["context"]
        ]
        kept = set(baseline_chunks) & set(reranked_chunks)
        promoted = set(reranked_chunks) - set(baseline_chunks)
        dropped = set(baseline_chunks) - set(reranked_chunks)

        print(f"Interleave top 5: {baseline_chunks}")
        print(f"Rerank top 5:     {reranked_chunks}")
        print(f"  Kept by both:   {len(kept)} chunks")
        print(f"  Promoted by rerank: {sorted(promoted)}")
        print(f"  Dropped by rerank:  {sorted(dropped)}")
        print()
        print(f"Answer (interleave):\n{baseline['answer']}\n")
        print(f"Answer (rerank):\n{reranked['answer']}\n")


if __name__ == "__main__":
    main()
