# ch08.2-rerank_basic.py - The same candidate pool, scored by a cross-encoder
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from reranker import rerank
from rag import _interleave

QUERY = "What happens to my unused vacation when I leave the company?"
NEEDLE = "unused vacation"


def show_top(label: str, chunks: list[dict], score_key: str | None = None):
    print(f"{label}:")
    print(f"  {'pos':<5}{'score':<10}source")
    print(f"  {'-'*5}{'-'*10}{'-'*50}")
    for i, c in enumerate(chunks, start=1):
        score = c.get(score_key, "") if score_key else ""
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else ""
        canonical = "*" if NEEDLE.lower() in c["content"].lower() else " "
        print(
            f" {canonical}{i:<5}{score_str:<10}"
            f"{c['source_file']}#chunk-{c['chunk_index']}"
        )
    print()


def main():
    candidates = hybrid_candidates(QUERY, k=20)
    print(f"Query: {QUERY!r}")
    print(f"Candidate pool: {len(candidates)} chunks  ('*' = chunk contains {NEEDLE!r})\n")

    interleaved = _interleave(candidates, n=5)
    show_top("Interleave (Chapter 7 default)", interleaved)

    reranked = rerank(QUERY, candidates, top_n=5)
    show_top("Cross-encoder rerank (bge-reranker-v2-m3)", reranked, score_key="rerank_score")

    canon_before = sum(1 for c in interleaved if NEEDLE.lower() in c["content"].lower())
    canon_after = sum(1 for c in reranked if NEEDLE.lower() in c["content"].lower())
    print(f"Canonical chunks in top 5: interleave={canon_before}  rerank={canon_after}")


if __name__ == "__main__":
    main()
