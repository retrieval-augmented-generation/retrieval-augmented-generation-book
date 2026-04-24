# corpus.py -- Sentences extracted from the Acme Corp knowledge base
#
# 30 sentences (5 per category) drawn from the Acme Corp corpus in data/corpus/.
# Each sentence is self-contained and uses vocabulary distinctive to its category,
# so that embedding clusters are clearly visible in the UMAP visualization.
# The full Acme Corp documents are ingested properly starting in chapter 5.
documents = [
    # Operations
    "Acme Corp guarantees 99.95% uptime for all API endpoints under the enterprise SLA.",
    "Enterprise customers may request a full refund within 45 days of purchase.",
    "Standard shipping takes 5 to 7 business days from the date the order is processed.",
    "The Pro tier pricing increased from $49 to $59 per seat per month effective November 2024.",
    "Enterprise provisioning from signed contract to go-live takes 10 to 15 business days.",

    # Engineering
    "The ML platform uses PostgreSQL with HNSW vector indexes for semantic search at sub-50ms latency.",
    "All production deployments use a blue-green model with canary rollout at 5% traffic.",
    "Code review requires at least one approving review from a code owner before merge.",
    "The feature store serves real-time features from Redis with a p99 read latency of 2 milliseconds.",
    "Python, Go, and TypeScript are the three approved languages across the engineering stack.",

    # IT & Security
    "Passwords must be at least 12 characters and rotated every 90 days per Section 4.2.1.",
    "All confidential data must be encrypted at rest using AES-256 and in transit using TLS 1.3.",
    "Audit logs are retained for 1 year in hot storage and 6 years in cold storage.",
    "Database backups target a recovery point objective of 1 hour and recovery time of 4 hours.",
    "Access controls follow the principle of least privilege with quarterly reviews by system owners.",

    # Product
    "API authentication supports both API keys and OAuth 2.0 authorization code flow.",
    "Webhooks deliver real-time event notifications to customer endpoints via HTTP POST.",
    "The REST API uses cursor-based pagination with a default page size of 25 results.",
    "Hybrid search combines BM25 keyword matching with semantic vector similarity for ranking.",
    "Account administrators manage users, configure SSO, and generate API keys in the admin console.",

    # Compliance
    "Customer data must be retained for a minimum of 7 years from the date of the last transaction.",
    "EU customer personal data must be deleted or anonymized within 3 years under GDPR rules.",
    "The HIPAA Security Rule requires access controls that restrict ePHI to authorized persons.",
    "Upon account termination all customer data including backups is deleted within 90 days.",
    "The maximum GDPR administrative fine is 4% of annual global turnover or 20 million euros.",

    # HR & Benefits
    "New employees accrue 15 vacation days in their first year, increasing to 20 days after year two.",
    "Remote work requires manager approval with a minimum of 2 days per week in the office.",
    "The annual performance review process includes quarterly check-ins and a 360-degree review.",
    "Expense reports over $500 require approval from both the direct manager and department head.",
    "Birthing parents receive 16 weeks of paid parental leave after 1 year of employment.",
]
