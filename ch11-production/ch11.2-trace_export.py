# ch11.2-trace_export.py - Export the per-request trace as one JSON file per request
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from rag import answer

TRACE_DIR = Path(__file__).parent / "traces"
QUERIES = [
    "What is the password rotation policy?",
    "What is the enterprise SLA uptime guarantee?",
]


def main():
    TRACE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for i, q in enumerate(QUERIES, start=1):
        result = answer(q)
        path = TRACE_DIR / f"{timestamp}-q{i}-{result['trace_id']}.json"
        path.write_text(json.dumps({
            "query": q,
            "trace": result["trace"],
            "answer": result["answer"],
        }, indent=2))
        print(f"Wrote {path.name}")

    print()
    print("In production, traces stream to an observability platform")
    print("(Jaeger, Honeycomb, Datadog, your-own-database).")
    print(f"For this demo, they live in {TRACE_DIR.name}/.")


if __name__ == "__main__":
    main()
