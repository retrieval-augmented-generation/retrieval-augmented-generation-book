# ch11.1-instrumented_pipeline.py - Run one query and print the full stage-level trace
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag import answer
from observability import trace_events

QUERY = "What is the bereavement leave policy?"


def main():
    print(f"Question: {QUERY}\n")
    result = answer(QUERY)

    print("Per-stage events:")
    for event in trace_events():
        # Drop a few high-volume fields from the printed view
        view = {k: v for k, v in event.items() if k not in ("ts", "trace_id")}
        print(json.dumps(view))

    print()
    print("Trace summary:")
    print(json.dumps(result["trace"], indent=2))

    print()
    print(f"Answer ({len(result['answer'])} chars):")
    print(result["answer"])


if __name__ == "__main__":
    main()
