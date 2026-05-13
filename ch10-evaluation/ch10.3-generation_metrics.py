# ch10.3-generation_metrics.py - Faithfulness, relevance, and deterministic citation accuracy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag import answer
from evaluation import (
    load_eval_set, faithfulness, relevance, citation_accuracy,
)


def main():
    eval_set = load_eval_set()
    print(f"Eval set: {len(eval_set)} queries\n")
    print(f"{'Query':<58}{'Faith':<8}{'Rel':<6}{'CiteAcc':<10}")
    print("-" * 82)

    faiths, rels, cites = [], [], []
    for item in eval_set:
        result = answer(item["query"], k=20, n=5, use_reranker=True)
        ans = result["answer"]
        ctx = result["context"]

        faith = faithfulness(item["query"], ctx, ans)
        rel = relevance(item["query"], ans, should_answer=item["should_answer"])
        cite = citation_accuracy(ans, ctx)
        faiths.append(faith)
        rels.append(rel)
        cites.append(cite)

        print(f"{item['query'][:56]:<58}{faith:<8.2f}{rel:<6.2f}{cite:<10.2f}")

    print()
    print(f"Mean faithfulness:       {sum(faiths) / len(faiths):.3f}")
    print(f"Mean relevance:          {sum(rels) / len(rels):.3f}")
    print(f"Mean citation accuracy:  {sum(cites) / len(cites):.3f}")


if __name__ == "__main__":
    main()
