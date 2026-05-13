# evaluation.py - Retrieval and generation metrics for RAG pipelines
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
JUDGE_MODEL = "gpt-4o-mini"

DEFAULT_EVAL_PATH = Path(__file__).parent / "eval_set.jsonl"

CITATION_MARKER = re.compile(r"\[([^\]#]+)#chunk-([^\]]+)\]")


def load_eval_set(path: Path = DEFAULT_EVAL_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recall_at_k(chunks: list[dict], canonical_sources: list[str]) -> float:
    """Fraction of canonical sources found among the supplied chunks."""
    if not canonical_sources:
        return 1.0  # Nothing to find; trivially perfect.
    sources_present = {c["source_file"] for c in chunks}
    hits = sum(1 for s in canonical_sources if s in sources_present)
    return hits / len(canonical_sources)


def first_canonical_rank(chunks: list[dict], canonical_sources: list[str]) -> int | None:
    """1-indexed rank of the first canonical chunk in the list, or None."""
    if not canonical_sources:
        return None
    canonical = set(canonical_sources)
    for i, c in enumerate(chunks, 1):
        if c["source_file"] in canonical:
            return i
    return None


def mrr(per_query_first_rank: list[int | None]) -> float:
    """Mean reciprocal rank. Queries with no canonical hit contribute 0."""
    if not per_query_first_rank:
        return 0.0
    total = sum(1.0 / r for r in per_query_first_rank if r is not None)
    return total / len(per_query_first_rank)


def citation_accuracy(answer: str, context_chunks: list[dict]) -> float:
    """Fraction of cited [file#chunk-N] markers that correspond to chunks in context.

    A marker that names a real file but a chunk index not present in context
    (e.g. an invented "#chunk-3.3") counts as invalid. Answers with no
    citation markers score 1.0 by default; refusals are therefore not penalised.
    """
    cited = CITATION_MARKER.findall(answer)
    if not cited:
        return 1.0
    in_context = {
        (c["source_file"], str(c["chunk_index"])) for c in context_chunks
    }
    valid = sum(1 for file, chunk in cited if (file, chunk) in in_context)
    return valid / len(cited)


def _judge(prompt: str) -> float:
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    # Accept "1.0", "0.5", "0.0" or first number found.
    for token in text.replace(",", " ").split():
        try:
            value = float(token)
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass
    return 0.0


FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to the source chunks it was given.

Question: {question}

Source chunks (the ONLY evidence the answer could legitimately use):
{context}

Answer: {answer}

Score the answer on faithfulness:
- 1.0 if every factual claim in the answer is supported by at least one source chunk
- 0.5 if the answer is partially supported (some claims supported, some not)
- 0.0 if the answer contradicts the sources, fabricates facts not in the sources, or uses external knowledge

If the answer is exactly "I don't know", treat it as faithful by default.

Respond with only a single number: 1.0, 0.5, or 0.0."""


RELEVANCE_PROMPT = """Does the answer directly address the question?

Question: {question}
Answer: {answer}

Score:
- 1.0 if the answer directly addresses the question
- 0.5 if the answer is partially relevant
- 0.0 if the answer is off-topic

Respond with only: 1.0, 0.5, or 0.0."""


def faithfulness(question: str, context_chunks: list[dict], answer: str) -> float:
    if answer.strip().lower().startswith("i don't know"):
        return 1.0
    ctx_text = "\n\n".join(
        f"[{c['source_file']}#chunk-{c['chunk_index']}] {c['content']}"
        for c in context_chunks
    )
    return _judge(FAITHFULNESS_PROMPT.format(
        question=question, context=ctx_text, answer=answer,
    ))


def relevance(question: str, answer: str, should_answer: bool = True) -> float:
    """Relevance is conditional on should_answer.

    If the corpus has no answer (should_answer=False), an exact refusal scores 1.0
    and anything else scores 0.0. If the corpus does have an answer, a refusal
    scores 0.0 and a non-refusal is judged by the LLM."""
    is_refusal = answer.strip().lower().startswith("i don't know")
    if not should_answer:
        return 1.0 if is_refusal else 0.0
    if is_refusal:
        return 0.0
    return _judge(RELEVANCE_PROMPT.format(question=question, answer=answer))
