# ch07.2-rag_pipeline.py - End-to-end worked example: query in, answer out
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag import answer

QUERY = "How do I get my money back?"


def main():
    print(f"Question: {QUERY}\n")
    result = answer(QUERY, k=20, n=5)
    print(f"Candidate pool size: {result['candidate_count']}")
    print(f"Chunks passed to LLM: {len(result['context'])}")
    print()
    print("Context sources:")
    for i, c in enumerate(result["context"], start=1):
        print(
            f"  {i}. {c['source_file']} (chunk {c['chunk_index']}, "
            f"matched_by={c['matched_by']})"
        )
    print()
    print("Answer:")
    print(result["answer"])


if __name__ == "__main__":
    main()
