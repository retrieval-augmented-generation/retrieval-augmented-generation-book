# ch06.4-side_by_side.py - Five query types through both retrievers, summary table
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import keyword_search, semantic_search


# Five canonical query shapes. Each one stresses a different retrieval pattern.
QUERIES = [
    ("Exact regulation",
     "HIPAA Section 164.312(a)(1)",
     "164.312(a)(1)"),
    ("Acronym / identifier",
     "CVE-2024-1234",
     "CVE-2024-1234"),
    ("Paraphrase",
     "How long do we keep customer information?",
     "retention"),
    ("Broad conceptual",
     "data retention rules",
     "retention"),
    ("Mixed (concept + identifier)",
     "HIPAA breach notification timeline",
     "breach"),
]


def found_in_top(results: list[dict], needle: str, top: int) -> bool:
    return any(
        needle.lower() in r["content"].lower()
        for r in results[:top]
    )


def main():
    print(f"{'Query type':<32}{'Keyword@5':<12}{'Semantic@5':<12}")
    print("-" * 56)
    for kind, query, needle in QUERIES:
        kw = keyword_search(query, k=5)
        sem = semantic_search(query, k=5)
        kw_ok = "yes" if found_in_top(kw, needle, 5) else "no"
        sem_ok = "yes" if found_in_top(sem, needle, 5) else "no"
        print(f"{kind:<32}{kw_ok:<12}{sem_ok:<12}")
    print()
    print("'yes' means the top-5 candidates from that retriever contain the")
    print("expected evidence string for the query.")


if __name__ == "__main__":
    main()
