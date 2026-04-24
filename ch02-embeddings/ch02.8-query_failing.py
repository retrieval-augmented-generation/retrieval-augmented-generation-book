# failing_query.py -- Queries that expose embedding limitations
from search import search, print_results

failing_queries = [
    "PDPA data residency rules for APAC",   # Partial match -- corpus has GDPR/HIPAA but not PDPA
    "CVE-2024-1234",                         # Exact identifier -- embeddings wash it out
    "policies NOT related to compliance",    # Negation -- embeddings ignore "NOT"
]

for query in failing_queries:
    results = search(query, top_k=3)
    print_results(query, results)
