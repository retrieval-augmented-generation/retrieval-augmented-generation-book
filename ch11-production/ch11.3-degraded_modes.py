# ch11.3-degraded_modes.py - Simulate stage outages and watch the pipeline degrade
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from rag import answer
from observability import trace_events

QUERY = "What is the bereavement leave policy?"


def run(label: str):
    print("=" * 70)
    print(label)
    print("=" * 70)
    result = answer(QUERY)

    print("Trace (stage and one-line summary):")
    for event in trace_events():
        stage = event["stage"]
        summary = {k: v for k, v in event.items()
                   if k not in ("ts", "trace_id", "stage")}
        print(f"  {stage:<28} {json.dumps(summary)}")
    print()
    print(f"Answer (first 200 chars): {result['answer'][:200]}")
    print()


def main():
    run("Healthy pipeline")

    config.FORCE_RERANKER_FAILURE = True
    run("Reranker outage (FORCE_RERANKER_FAILURE=true)  -> RRF fallback")
    config.FORCE_RERANKER_FAILURE = False

    config.FORCE_KEYWORD_FAILURE = True
    run("Keyword retrieval outage  -> semantic-only candidate pool")
    config.FORCE_KEYWORD_FAILURE = False


if __name__ == "__main__":
    main()
