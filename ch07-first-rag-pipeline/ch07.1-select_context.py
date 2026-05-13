# ch07.1-select_context.py - Three context selection strategies on one candidate pool
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from rag import select_context

QUERY = "What is the HIPAA Section 164.312(a)(1) requirement?"
N = 5


def show(label: str, chunks: list[dict]) -> None:
    print(f"{label}:")
    for c in chunks:
        kw = c["keyword_rank"] or "-"
        sem = c["semantic_rank"] or "-"
        print(
            f"  kw={kw:<3}  sem={sem:<3}  {c['matched_by']:<8}  "
            f"{c['source_file']} (chunk {c['chunk_index']})"
        )
    print()


def main():
    candidates = hybrid_candidates(QUERY, k=20)
    print(f"Query: {QUERY!r}")
    print(f"Candidate pool: {len(candidates)} chunks\n")

    show("Strategy: interleave (default)", select_context(candidates, n=N, strategy="interleave"))
    show("Strategy: keyword (top by keyword rank)", select_context(candidates, n=N, strategy="keyword"))
    show("Strategy: semantic (top by semantic rank)", select_context(candidates, n=N, strategy="semantic"))


if __name__ == "__main__":
    main()
