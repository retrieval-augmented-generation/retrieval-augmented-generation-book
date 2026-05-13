# ch10.2-retrieval_metrics.py - Candidate recall vs final-context recall, plus MRR
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from rag import _interleave
from evaluation import load_eval_set, recall_at_k, first_canonical_rank, mrr


K_POOL = 20
N_CONTEXT = 5


def main():
    eval_set = load_eval_set()
    print(f"Eval set: {len(eval_set)} queries\n")

    print(f"CandR@{K_POOL} = candidate recall at K  "
          f"CtxR@{N_CONTEXT} = final-context recall at N  "
          f"Rank = first canonical chunk rank in pool")
    print(f"{'Query':<60}{'CandR@'+str(K_POOL):<10}{'CtxR@'+str(N_CONTEXT):<10}{'Rank':<6}")
    print("-" * 86)

    pool_recalls = []
    context_recalls = []
    first_ranks = []

    for item in eval_set:
        if not item["should_answer"]:
            continue  # Recall on unanswerable queries is not meaningful.
        candidates = hybrid_candidates(item["query"], k=K_POOL)
        selected = _interleave(candidates, N_CONTEXT)

        pool_r = recall_at_k(candidates, item["canonical_sources"])
        ctx_r = recall_at_k(selected, item["canonical_sources"])
        rank = first_canonical_rank(candidates, item["canonical_sources"])

        pool_recalls.append(pool_r)
        context_recalls.append(ctx_r)
        first_ranks.append(rank)

        rank_str = str(rank) if rank else ">K"
        print(f"{item['query'][:58]:<60}{pool_r:<10.2f}{ctx_r:<10.2f}{rank_str:<6}")

    print()
    print(f"Mean Candidate Recall@{K_POOL}:      "
          f"{sum(pool_recalls) / len(pool_recalls):.3f}")
    print(f"Mean Final-context Recall@{N_CONTEXT}:  "
          f"{sum(context_recalls) / len(context_recalls):.3f}")
    print(f"MRR over candidate pool:        {mrr(first_ranks):.3f}")


if __name__ == "__main__":
    main()
