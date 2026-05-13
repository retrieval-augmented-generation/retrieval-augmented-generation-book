# ch07.3-five_queries.py - Run five Acme Corp queries through the full pipeline
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag import answer

QUERIES = [
    "What multi-factor authentication methods does Acme Corp support?",
    "When does open enrollment start and what are the plan options?",
    "What is Acme Corp's data retention policy for customer records?",
    "What is the bereavement leave policy?",
    "What is Acme Corp's cryptocurrency policy?",
]


def cited_markers(text: str) -> list[tuple[str, int]]:
    """Parse [file#chunk-N] markers from the answer text."""
    return [
        (file, int(chunk))
        for file, chunk in re.findall(r"\[([^\]#]+)#chunk-(\d+)\]", text)
    ]


def main():
    for i, q in enumerate(QUERIES, start=1):
        print("=" * 70)
        print(f"Q{i}: {q}")
        print("=" * 70)
        result = answer(q, k=20, n=5)

        print(f"Pool size: {result['candidate_count']} candidates")
        print("Chunks in context:")
        for c in result["context"]:
            print(
                f"  {c['source_file']}#chunk-{c['chunk_index']}  "
                f"(matched_by={c['matched_by']})"
            )

        cites = cited_markers(result["answer"])
        shown = {(c["source_file"], c["chunk_index"]) for c in result["context"]}
        same_file_diff_chunk = [
            (f, n) for (f, n) in cites
            if (f, n) not in shown and any(c["source_file"] == f for c in result["context"])
        ]
        unknown_file = [
            (f, n) for (f, n) in cites
            if not any(c["source_file"] == f for c in result["context"])
        ]
        print(f"Markers cited in answer: {cites}")
        if unknown_file:
            print(f"Citations to FILES not in context: {unknown_file}")
        if same_file_diff_chunk:
            print(f"Citations to file-in-context but DIFFERENT chunk: {same_file_diff_chunk}")
        print()
        print(f"Answer:\n{result['answer']}\n")


if __name__ == "__main__":
    main()
