# run_queries.py -- Test the search function with various queries
from search import search, print_results

queries = [
    "How do I get my money back?",
    "What is the SLA uptime guarantee?",
    "How does the deployment pipeline work?",
    "What are the password requirements?",
    "How do I authenticate API requests?",
    "How many vacation days do new employees get?",
]

for query in queries:
    results = search(query)
    print_results(query, results)
