# rag.py - End-to-end RAG: query in, grounded answer + cited sources out
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from openai import OpenAI

from hybrid import hybrid_candidates

load_dotenv()
client = OpenAI()

LLM_MODEL = "gpt-4o-mini"

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


def select_context(
    candidates: list[dict], n: int = 5, strategy: str = "interleave"
) -> list[dict]:
    """Pick top-N chunks from a candidate pool. Default strategy: interleave."""
    if strategy == "interleave":
        return _interleave(candidates, n)
    if strategy == "keyword":
        ranked = [c for c in candidates if c["keyword_rank"] is not None]
        ranked.sort(key=lambda c: c["keyword_rank"])
        return ranked[:n]
    if strategy == "semantic":
        ranked = [c for c in candidates if c["semantic_rank"] is not None]
        ranked.sort(key=lambda c: c["semantic_rank"])
        return ranked[:n]
    raise ValueError(f"unknown selection strategy: {strategy}")


def _interleave(candidates: list[dict], n: int) -> list[dict]:
    """Alternate keyword[i] and semantic[i], deduplicating by chunk_id."""
    kw = sorted(
        [c for c in candidates if c["keyword_rank"] is not None],
        key=lambda c: c["keyword_rank"],
    )
    sem = sorted(
        [c for c in candidates if c["semantic_rank"] is not None],
        key=lambda c: c["semantic_rank"],
    )

    selected: list[dict] = []
    seen: set[int] = set()
    for kw_c, sem_c in zip(kw, sem):
        for c in (kw_c, sem_c):
            if c["chunk_id"] not in seen:
                selected.append(c)
                seen.add(c["chunk_id"])
            if len(selected) >= n:
                return selected

    # If one list ran out before we hit n, fill from whichever side has more.
    for c in kw + sem:
        if c["chunk_id"] not in seen:
            selected.append(c)
            seen.add(c["chunk_id"])
        if len(selected) >= n:
            break

    return selected


def build_prompt(question: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        marker = f"[{c['source_file']}#chunk-{c['chunk_index']}]"
        blocks.append(f"{i}. {marker} {c['content']}")
    context = "\n\n".join(blocks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def answer(
    question: str, k: int = 20, n: int = 5, strategy: str = "interleave"
) -> dict:
    candidates = hybrid_candidates(question, k=k)
    chunks = select_context(candidates, n=n, strategy=strategy)
    prompt = build_prompt(question, chunks)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return {
        "answer": response.choices[0].message.content.strip(),
        "context": chunks,
        "candidate_count": len(candidates),
    }
