# ch13.1-tool_call.py - Single-shot agent: the model decides to call search once
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import agent_answer

QUERY = "What is the bereavement leave policy?"


def main():
    print(f"Question: {QUERY}\n")
    result = agent_answer(QUERY, max_steps=4, max_search_calls=3)

    print(f"Agent steps:        {result['steps']}")
    print(f"Search calls made:  {result['search_calls']}")
    print()
    print("Agent decision trace:")
    for event in result["events"]:
        stage = event["stage"]
        summary = {k: v for k, v in event.items()
                   if k not in ("ts", "trace_id", "stage")}
        print(f"  {stage:<26} {summary}")
    print()
    print("Answer:")
    print(result["answer"])


if __name__ == "__main__":
    main()
