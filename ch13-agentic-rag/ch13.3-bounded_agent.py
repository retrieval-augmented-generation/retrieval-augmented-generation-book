# ch13.3-bounded_agent.py - Step and search-budget limits in an agentic loop
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import agent_answer

# Deliberately open-ended question that may tempt the agent into many searches.
QUERY = (
    "Give me a complete overview of every Acme Corp policy that touches "
    "customer data, including retention, deletion, transfer, classification, "
    "encryption, audit, and breach notification."
)


def run(label: str, max_steps: int, max_search_calls: int):
    print("=" * 70)
    print(f"{label}  (max_steps={max_steps}, max_search_calls={max_search_calls})")
    print("=" * 70)
    result = agent_answer(
        QUERY, max_steps=max_steps, max_search_calls=max_search_calls,
    )
    successful = sum(1 for e in result["events"] if e["stage"] == "agent.tool.result")
    refused = sum(1 for e in result["events"] if e["stage"] == "agent.tool.refused")
    print(f"Steps taken:              {result['steps']}")
    print(f"Successful search calls:  {successful}")
    print(f"Refused tool calls:       {refused}")
    print(f"Attempted tool calls:     {successful + refused}")
    print()
    print("Tool call summary:")
    for event in result["events"]:
        if event["stage"] in ("agent.tool.result", "agent.tool.refused"):
            stage = event["stage"]
            info = {k: v for k, v in event.items()
                    if k not in ("ts", "trace_id", "stage")}
            print(f"  {stage:<22} {info}")
    print()
    print(f"Answer (first 300 chars):\n{result['answer'][:300]}")
    print()


def main():
    print(f"Question: {QUERY!r}\n")
    run("Tight bound",   max_steps=3, max_search_calls=2)
    run("Looser bound",  max_steps=6, max_search_calls=5)


if __name__ == "__main__":
    main()
