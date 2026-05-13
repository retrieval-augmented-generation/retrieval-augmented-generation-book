# agent.py - Tool-using RAG agent: the LLM decides when and what to retrieve
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))

import config
from hybrid import hybrid_candidates
from observability import log_event, new_trace, span, trace_events
from reranker import rerank


load_dotenv()
client = OpenAI()


SYSTEM_PROMPT = """You are an Acme Corp internal assistant. To answer the user's question, call the `search` tool to retrieve relevant chunks from the corporate knowledge base. You may call `search` multiple times with different queries if you need different pieces of evidence.

When you have enough evidence to answer:
- Use only the chunks returned by your search calls. Do not rely on outside knowledge.
- Cite each fact with the exact chunk marker from the search results, e.g. [hipaa_compliance_overview.md#chunk-6].
- If the evidence is insufficient, reply exactly: "I don't know."

Do not call `search` after you have produced a final answer."""


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search Acme Corp's internal knowledge base. Returns up to k chunks "
            "with source identifiers and content excerpts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in natural language",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of chunks to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def _do_search(query: str, k: int) -> list[dict]:
    """The actual retrieval the agent's tool call resolves to."""
    candidates = hybrid_candidates(query, k=20)
    return rerank(query, candidates, top_n=k)


def _serialise_chunks(chunks: list[dict]) -> str:
    """Compact representation passed back to the LLM as the tool result."""
    return json.dumps([
        {
            "marker": f"{c['source_file']}#chunk-{c['chunk_index']}",
            "content": c["content"][:600],
        }
        for c in chunks
    ])


def agent_answer(
    question: str,
    max_steps: int = 5,
    max_search_calls: int = 4,
) -> dict:
    """Run the agent loop. Returns answer + step count + trace summary."""
    trace_id = new_trace()
    log_event(
        "agent.start", query=question, max_steps=max_steps,
        max_search_calls=max_search_calls,
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    search_calls_made = 0
    last_chunks: list[dict] = []

    for step in range(1, max_steps + 1):
        with span(f"agent.step.{step}.llm"):
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=[SEARCH_TOOL],
                temperature=0,
            )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            log_event(
                "agent.final",
                step=step, search_calls_made=search_calls_made,
                output_tokens=response.usage.completion_tokens,
            )
            return {
                "trace_id": trace_id,
                "answer": msg.content,
                "steps": step,
                "search_calls": search_calls_made,
                "events": trace_events(),
                "last_chunks": last_chunks,
            }

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            search_calls_made += 1
            if search_calls_made > max_search_calls:
                tool_text = json.dumps({"error": "search call budget exhausted"})
                log_event(
                    "agent.tool.refused",
                    step=step, query=args.get("query"),
                    reason="search_budget_exhausted",
                )
            else:
                with span(f"agent.step.{step}.search",
                          query=args.get("query"), k=args.get("k", 5)):
                    chunks = _do_search(args.get("query", ""), args.get("k", 5))
                last_chunks = chunks
                tool_text = _serialise_chunks(chunks)
                log_event(
                    "agent.tool.result",
                    step=step, query=args.get("query"),
                    chunk_count=len(chunks),
                )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_text,
            })

    log_event("agent.step_budget_exhausted", step=max_steps)
    return {
        "trace_id": trace_id,
        "answer": "I reached the step budget without finishing.",
        "steps": max_steps,
        "search_calls": search_calls_made,
        "events": trace_events(),
        "last_chunks": last_chunks,
    }
