# ch10.1-bootstrap_eval_set.py - Generate candidate eval questions from random chunks.
# Output is a JSONL of (chunk, question) pairs for human curation. Do not use the
# raw LLM output as a labeled eval set without manual review.
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from openai import OpenAI

from database import get_connection

load_dotenv()
client = OpenAI()
LLM_MODEL = "gpt-4o-mini"
N_CHUNKS = 5

PROMPT = """Generate one specific question that this chunk of an internal Acme Corp document would directly answer. The question should be the kind of question an employee would naturally ask, not a paraphrase of the chunk's first sentence. Return only the question.

Chunk source: {source}
Chunk content:
{content}

Question:"""


def main():
    random.seed(42)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, content, source_file, chunk_index
            FROM chunks
            WHERE length(content) > 400
            ORDER BY random()
            LIMIT %s
            """,
            (N_CHUNKS,),
        )
        rows = cur.fetchall()
    conn.close()

    print(f"Bootstrapping {len(rows)} candidate questions from random chunks.\n")
    output = []
    for chunk_id, content, source, chunk_index in rows:
        prompt = PROMPT.format(source=source, content=content[:1500])
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        question = response.choices[0].message.content.strip()

        record = {
            "candidate_question": question,
            "canonical_sources": [source],
            "source_chunk": f"{source}#chunk-{chunk_index}",
            "should_answer": True,
            "_review_status": "unreviewed",
        }
        output.append(record)

        print(f"Q: {question}")
        print(f"   from {source}#chunk-{chunk_index}\n")

    out_path = Path(__file__).parent / "bootstrap_candidates.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in output) + "\n")
    print(f"Wrote {len(output)} candidates to {out_path.name}")
    print("Review each candidate, edit as needed, then promote to eval_set.jsonl.")


if __name__ == "__main__":
    main()
