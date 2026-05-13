# ch06.6-filtered_candidates.py - Metadata filters as candidate-set scoping
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hybrid import hybrid_candidates
from database import get_connection

QUERY = "How long do we keep customer information?"


def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def main():
    print(f"Query: {QUERY!r}\n")

    unfiltered, t_un = time_call(hybrid_candidates, QUERY, k=20)
    print(f"No filter:              {len(unfiltered):>3} candidates  ({t_un:6.1f} ms)")

    filtered, t_f = time_call(
        hybrid_candidates, QUERY, k=20, filters={"category": "Compliance"}
    )
    print(f"category=Compliance:    {len(filtered):>3} candidates  ({t_f:6.1f} ms)")

    pdf_only, t_p = time_call(
        hybrid_candidates, QUERY, k=20, filters={"doc_type": "pdf"}
    )
    print(f"doc_type=pdf:           {len(pdf_only):>3} candidates  ({t_p:6.1f} ms)")
    print()

    # Post-filtering baseline: fetch everything, drop non-matching in Python.
    full, _ = time_call(hybrid_candidates, QUERY, k=200)
    start = time.perf_counter()
    post = [c for c in full if c.get("category") == "Compliance"]
    t_post = (time.perf_counter() - start) * 1000
    print(
        f"Post-filter in Python:  fetched {len(full)} candidates "
        f"then filtered to {len(post)} ({t_post:.1f} ms after fetch)"
    )
    print()

    # EXPLAIN ANALYZE for the SQL-level pre-filtered keyword query.
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            EXPLAIN ANALYZE
            SELECT id
            FROM chunks
            WHERE search_vector @@ websearch_to_tsquery('english', %s)
              AND category = %s
            ORDER BY ts_rank_cd(
                search_vector, websearch_to_tsquery('english', %s)
            ) DESC
            LIMIT 20
            """,
            (QUERY, "Compliance", QUERY),
        )
        plan = "\n".join(row[0] for row in cur.fetchall())
    conn.close()

    print("EXPLAIN ANALYZE for pre-filtered keyword query:")
    print(plan)


if __name__ == "__main__":
    main()
