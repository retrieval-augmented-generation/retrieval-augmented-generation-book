# ch13.2-multi_step.py - A question that requires multiple search calls
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import agent_answer

QUERY = (
    "Compare bereavement leave and parental leave at Acme Corp. "
    "How many paid days does each give, and who is eligible?"
)


def main():
    print(f"Question: {QUERY}\n")
    result = agent_answer(QUERY, max_steps=6, max_search_calls=4)

    print(f"Agent steps:        {result['steps']}")
    print(f"Search calls made:  {result['search_calls']}")
    print()
    print("Tool calls the agent decided to issue:")
    for event in result["events"]:
        if event["stage"] == "agent.tool.result":
            print(f"  step {event['step']}: query={event['query']!r}  "
                  f"chunks={event['chunk_count']}")
    print()
    print("Answer:")
    print(result["answer"])


if __name__ == "__main__":
    main()
