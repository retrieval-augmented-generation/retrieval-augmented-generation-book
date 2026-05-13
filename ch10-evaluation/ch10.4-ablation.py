# ch10.4-ablation.py - Compare four pipeline configurations on the same eval set
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from openai import OpenAI

from hybrid import hybrid_candidates, semantic_search
from reranker import rerank
from rag import _interleave, build_prompt
from query_transform import rewrite
from evaluation import (
    load_eval_set, recall_at_k, first_canonical_rank, mrr,
    faithfulness, relevance, citation_accuracy,
)

load_dotenv()
client = OpenAI()
LLM_MODEL = "gpt-4o-mini"

K_POOL = 20
N_CONTEXT = 5

OUTPUT_PATH = Path(__file__).parent / "ablation_results.json"


def select_semantic_only(query: str) -> tuple[list[dict], list[dict]]:
    """Pool = semantic top K. Context = top N semantic. No hybrid keyword side."""
    candidates = semantic_search(query, k=K_POOL)
    return candidates, candidates[:N_CONTEXT]


def select_hybrid_interleave(query: str) -> tuple[list[dict], list[dict]]:
    """Pool = hybrid candidates. Context = interleave top N."""
    candidates = hybrid_candidates(query, k=K_POOL)
    return candidates, _interleave(candidates, N_CONTEXT)


def select_hybrid_rerank(query: str) -> tuple[list[dict], list[dict]]:
    """Pool = hybrid candidates. Context = reranker top N."""
    candidates = hybrid_candidates(query, k=K_POOL)
    return candidates, rerank(query, candidates, top_n=N_CONTEXT)


def select_full_pipeline(query: str) -> tuple[list[dict], list[dict]]:
    """Pool = hybrid candidates over rewritten query. Context = reranker top N."""
    rewritten = rewrite(query)
    candidates = hybrid_candidates(rewritten, k=K_POOL)
    return candidates, rerank(rewritten, candidates, top_n=N_CONTEXT)


def generate(question: str, chunks: list[dict]) -> str:
    prompt = build_prompt(question, chunks)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


CONFIGS = [
    ("Semantic only", select_semantic_only),
    ("Hybrid interleave", select_hybrid_interleave),
    ("Hybrid + rerank", select_hybrid_rerank),
    ("Full + rewrite", select_full_pipeline),
]


def evaluate_config(config_name: str, config_fn, eval_set: list[dict]) -> dict:
    pool_recalls, ctx_recalls = [], []
    pool_ranks, ctx_ranks = [], []
    faiths, rels, cites = [], [], []
    latencies = []
    for item in eval_set:
        t0 = time.perf_counter()
        candidates, context = config_fn(item["query"])
        ans = generate(item["query"], context)
        latency = time.perf_counter() - t0

        if item["should_answer"]:
            pool_recalls.append(recall_at_k(candidates, item["canonical_sources"]))
            ctx_recalls.append(recall_at_k(context, item["canonical_sources"]))
            pool_ranks.append(first_canonical_rank(candidates, item["canonical_sources"]))
            ctx_ranks.append(first_canonical_rank(context, item["canonical_sources"]))

        faiths.append(faithfulness(item["query"], context, ans))
        rels.append(relevance(item["query"], ans, should_answer=item["should_answer"]))
        cites.append(citation_accuracy(ans, context))
        latencies.append(latency)

    return {
        "config": config_name,
        "candidate_recall": sum(pool_recalls) / len(pool_recalls) if pool_recalls else None,
        "context_recall": sum(ctx_recalls) / len(ctx_recalls) if ctx_recalls else None,
        "pool_mrr": mrr(pool_ranks) if pool_ranks else None,
        "rerank_mrr": mrr(ctx_ranks) if ctx_ranks else None,
        "faithfulness": sum(faiths) / len(faiths),
        "relevance": sum(rels) / len(rels),
        "citation_accuracy": sum(cites) / len(cites),
        "avg_latency_s": sum(latencies) / len(latencies),
    }


def main():
    eval_set = load_eval_set()
    print(f"Eval set: {len(eval_set)} queries  "
          f"({sum(1 for i in eval_set if i['should_answer'])} answerable)\n")

    rows = []
    for name, fn in CONFIGS:
        print(f"Running {name}...")
        row = evaluate_config(name, fn, eval_set)
        rows.append(row)

    print()
    print("Retrieval metrics  (CandR@20 = candidate recall, CtxR@5 = final-context recall,")
    print("                   PoolMRR = MRR over candidate pool, RerankMRR = MRR over final context)")
    print(f"{'Config':<22}{'CandR@20':<10}{'CtxR@5':<10}{'PoolMRR':<10}{'RerankMRR':<11}{'Latency':<10}")
    print("-" * 73)
    for r in rows:
        cand_r = f"{r['candidate_recall']:.2f}" if r["candidate_recall"] is not None else "-"
        ctx_r = f"{r['context_recall']:.2f}" if r["context_recall"] is not None else "-"
        pool_mrr = f"{r['pool_mrr']:.2f}" if r["pool_mrr"] is not None else "-"
        rerank_mrr = f"{r['rerank_mrr']:.2f}" if r["rerank_mrr"] is not None else "-"
        print(
            f"{r['config']:<22}{cand_r:<10}{ctx_r:<10}"
            f"{pool_mrr:<10}{rerank_mrr:<11}{r['avg_latency_s']:<10.1f}"
        )

    print()
    print("Generation metrics  (Faith = faithfulness, Rel = relevance,")
    print("                    CiteAcc = deterministic citation accuracy)")
    print(f"{'Config':<22}{'Faith':<8}{'Rel':<6}{'CiteAcc':<10}")
    print("-" * 46)
    for r in rows:
        print(
            f"{r['config']:<22}{r['faithfulness']:<8.2f}{r['relevance']:<6.2f}"
            f"{r['citation_accuracy']:<10.2f}"
        )

    OUTPUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
