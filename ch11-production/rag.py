# rag.py - End-to-end RAG with stage-level observability and graceful degradation
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI

import config
from hybrid import hybrid_candidates
from observability import log_event, new_trace, span, trace_summary
from query_transform import rewrite
from reranker import rerank as cross_encoder_rerank

client = OpenAI()

PROMPT_TEMPLATE = """You are an Acme Corp internal assistant. Answer the question using only the context below.

Rules:
- Use only the information in the numbered context blocks. Do not rely on outside knowledge.
- Cite each fact with the exact source marker shown in brackets, including the chunk identifier, e.g. [hipaa_compliance_overview.md#chunk-6].
- Only cite markers that appear in the context below. Do not cite a document at file level; the chunk identifier is required.
- If the context does not contain enough information to answer the question, reply exactly: "I don't know."

Context:
{context}

Question: {question}

Answer:"""

CITATION_MARKER = re.compile(r"\[([^\]#]+)#chunk-([^\]]+)\]")
RRF_K = 60


def rrf_select(candidates: list[dict], n: int) -> list[dict]:
    """Reciprocal Rank Fusion fallback used when the cross-encoder is unavailable."""
    scored = []
    for c in candidates:
        score = 0.0
        if c["keyword_rank"] is not None:
            score += 1.0 / (RRF_K + c["keyword_rank"])
        if c["semantic_rank"] is not None:
            score += 1.0 / (RRF_K + c["semantic_rank"])
        scored.append(dict(c, rerank_score=score))
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:n]


def select_context(query: str, candidates: list[dict], n: int) -> list[dict]:
    """Reranker with RRF fallback. Logs which path was taken."""
    if not config.USE_RERANKER or not candidates:
        with span("select.rrf_fallback", reason="reranker_disabled"):
            return rrf_select(candidates, n)

    try:
        with span("select.rerank",
                  input_count=len(candidates), output_count=min(n, len(candidates))):
            if config.FORCE_RERANKER_FAILURE:
                raise RuntimeError("Simulated reranker outage")
            return cross_encoder_rerank(query, candidates, top_n=n)
    except Exception as e:
        log_event("select.rerank.degraded", error=repr(e))
        with span("select.rrf_fallback", reason="reranker_failure"):
            return rrf_select(candidates, n)


def build_prompt(question: str, chunks: list[dict]) -> str:
    blocks = [
        f"{i}. [{c['source_file']}#chunk-{c['chunk_index']}] {c['content']}"
        for i, c in enumerate(chunks, start=1)
    ]
    return PROMPT_TEMPLATE.format(context="\n\n".join(blocks), question=question)


def citation_accuracy(answer: str, chunks: list[dict]) -> tuple[float, int, int]:
    """Returns (accuracy, valid_count, total_count) for cited markers."""
    cited = CITATION_MARKER.findall(answer)
    if not cited:
        return 1.0, 0, 0
    in_context = {(c["source_file"], str(c["chunk_index"])) for c in chunks}
    valid = sum(1 for file, chunk in cited if (file, chunk) in in_context)
    return valid / len(cited), valid, len(cited)


def answer(question: str) -> dict:
    """Run the full pipeline on a single question. Emits a trace to observability."""
    trace_id = new_trace()
    log_event("request.start", query=question, **config.snapshot())

    if config.USE_QUERY_TRANSFORM:
        with span("query_transform.rewrite", method="rewrite"):
            retrieval_query = rewrite(question)
        log_event(
            "query_transform.applied",
            original=question, rewritten=retrieval_query,
        )
    else:
        retrieval_query = question

    candidates = hybrid_candidates(retrieval_query, k=config.CANDIDATE_K)
    selected = select_context(retrieval_query, candidates, n=config.CONTEXT_N)

    with span("generation.llm_call", context_count=len(selected)):
        prompt = build_prompt(question, selected)
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer_text = response.choices[0].message.content.strip()
        usage = response.usage

    cite_acc, valid_cites, total_cites = citation_accuracy(answer_text, selected)
    log_event(
        "generation.complete",
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        citation_count=total_cites,
        citation_valid=valid_cites,
        citation_accuracy=round(cite_acc, 3),
    )

    return {
        "trace_id": trace_id,
        "answer": answer_text,
        "context": selected,
        "candidate_count": len(candidates),
        "trace": trace_summary(),
    }
