# ch06.2-add_search_vector.py - Add a tsvector column and GIN index to chunks
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from database import get_connection


SAMPLE_QUERY = "HIPAA Section 164.312"


def explain(cur, sql: str, params: tuple) -> str:
    cur.execute("EXPLAIN ANALYZE " + sql, params)
    return "\n".join(row[0] for row in cur.fetchall())


def main():
    conn = get_connection()
    cur = conn.cursor()

    # 1) Baseline: query without a search_vector column.
    # This computes to_tsvector on every row at query time.
    print("=" * 60)
    print("BEFORE: full-text search without a stored tsvector or index")
    print("=" * 60)
    inline_sql = """
        SELECT id
        FROM chunks
        WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', %s)
        LIMIT 10
    """
    plan = explain(cur, inline_sql, (SAMPLE_QUERY,))
    print(plan)
    print()

    # 2) Add the column, populate it, build the GIN index. Idempotent.
    print("=" * 60)
    print("MIGRATION: adding search_vector column and GIN index")
    print("=" * 60)

    cur.execute(
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector"
    )
    cur.execute(
        """
        UPDATE chunks
        SET search_vector = to_tsvector('english', content)
        WHERE search_vector IS NULL
        """
    )
    cur.execute("SELECT count(*) FROM chunks WHERE search_vector IS NOT NULL")
    populated = cur.fetchone()[0]
    print(f"search_vector populated for {populated} chunks")

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS chunks_search_vector_idx
        ON chunks USING GIN (search_vector)
        """
    )
    print("GIN index chunks_search_vector_idx is in place")
    conn.commit()
    print()

    # 3) Indexed query plan.
    print("=" * 60)
    print("AFTER: full-text search using the GIN-indexed tsvector column")
    print("=" * 60)
    indexed_sql = """
        SELECT id
        FROM chunks
        WHERE search_vector @@ websearch_to_tsquery('english', %s)
        LIMIT 10
    """
    plan = explain(cur, indexed_sql, (SAMPLE_QUERY,))
    print(plan)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
