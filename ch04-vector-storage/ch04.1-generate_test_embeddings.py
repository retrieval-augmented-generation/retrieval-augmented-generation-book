# generate_test_embeddings.py
# Generate benchmark embeddings from the Acme Corp corpus.
#
# The Acme Corp knowledge base produces ~5-8K chunks after splitting.
# To benchmark vector indexing at enterprise scale (where exact search
# becomes impractical), we augment to 100K vectors by adding copies with
# small Gaussian noise. This preserves the semantic structure of the real
# embeddings while simulating a larger corpus. In production, 100K chunks
# is typical for a company with thousands of internal documents.

import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
OUTPUT_DIR = Path(__file__).resolve().parent

TARGET_VECTORS = 100_000
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
DIMENSION = 384  # all-MiniLM-L6-v2
NUM_QUERIES = 100

np.random.seed(42)

# ---------------------------------------------------------------------------
# Step 1: Load documents from the Acme Corp corpus
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT / "ch03-chunking"))
from chunkers import chunk_recursive


def load_document(path: Path) -> str:
    """Read a document and return its text content."""
    if path.suffix == ".pdf":
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    return path.read_text(errors="replace")


print("Loading Acme Corp corpus...")
corpus_files = sorted(
    p for p in CORPUS_DIR.iterdir()
    if p.is_file() and not p.name.startswith(".") and p.suffix in (
        ".pdf", ".md", ".html", ".docx", ".txt"
    )
)
print(f"  Found {len(corpus_files)} documents")

# ---------------------------------------------------------------------------
# Step 2: Chunk all documents
# ---------------------------------------------------------------------------
print("Chunking documents...")
chunks = []
chunk_metadata = []

for filepath in corpus_files:
    text = load_document(filepath)
    doc_chunks = chunk_recursive(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    doc_chunk_idx = 0
    for chunk_text in doc_chunks:
        if len(chunk_text.strip()) > 20:
            chunks.append(chunk_text)
            chunk_metadata.append({
                "source": filepath.name,
                "chunk_index": doc_chunk_idx,
            })
            doc_chunk_idx += 1

num_real = len(chunks)
print(f"  Produced {num_real} chunks from {len(corpus_files)} documents")

# ---------------------------------------------------------------------------
# Step 3: Embed all chunks
# ---------------------------------------------------------------------------
print("Embedding chunks with all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")
real_embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
real_embeddings = real_embeddings.astype("float32")
print(f"  Real embeddings shape: {real_embeddings.shape}")

# ---------------------------------------------------------------------------
# Step 4: Scale up to 100K with augmented copies
# ---------------------------------------------------------------------------
# The real embeddings preserve the semantic structure of the Acme Corp corpus.
# We augment to 100K by adding copies with small Gaussian noise (std=0.01).
# The noise is small enough that augmented vectors stay near their source,
# preserving cluster structure and making recall benchmarks meaningful.
# The first num_real vectors are always the originals.

print(f"Augmenting from {num_real} to {TARGET_VECTORS} vectors...")
augmented = [real_embeddings]
copies_needed = TARGET_VECTORS - num_real

while copies_needed > 0:
    batch_size = min(copies_needed, num_real)
    indices = np.random.choice(num_real, batch_size, replace=True)
    base = real_embeddings[indices]
    noise = np.random.normal(0, 0.01, base.shape).astype("float32")
    noisy = base + noise
    norms = np.linalg.norm(noisy, axis=1, keepdims=True)
    noisy = noisy / norms
    augmented.append(noisy)
    copies_needed -= batch_size

embeddings = np.vstack(augmented)[:TARGET_VECTORS].astype("float32")
print(f"  Final embeddings shape: {embeddings.shape}")

# ---------------------------------------------------------------------------
# Step 5: Generate query vectors from real book questions
# ---------------------------------------------------------------------------
base_queries = [
    # From chapter 7 evaluation set
    "What is the standard probationary period for new hires?",
    "What is the company's policy on jury duty leave?",
    "How many vacation days do employees get in their first year?",
    "What is the deadline for open enrollment?",
    "Who approves expense reports over $500?",
    "What are the differences between the three health plan tiers?",
    "What steps are required to request FMLA leave?",
    "How does the performance review process work?",
    "What are the requirements for the employee referral bonus?",
    "Compare the parental leave policies for birthing and non-birthing parents.",
    "What does Section 4.2.1 of the IT security policy say about password requirements?",
    "What is the RSU vesting schedule for Level 6 engineers?",
    "What is the maximum 401(k) employer match percentage?",
    "What are the rules for working from home?",
    "Tell me about the harassment policy and reporting procedures.",
    # From chapter 6 retrieval comparison
    "How do access controls protect sensitive data?",
    "What happens when a system fails to authenticate?",
    "What are the technical safeguards required for ePHI?",
    "What is the data retention policy?",
    "What are the SLA guarantees for API uptime?",
    # Additional queries
    "How does the deployment pipeline work?",
    "What security certifications does Acme Corp hold?",
    "What is the refund policy for enterprise customers?",
    "How do I reset my password?",
    "What are the backup and recovery targets?",
]

paraphrase_variants = [
    "How long is the probation period for new employees?",
    "Tell me about jury duty leave.",
    "What's the vacation accrual for year one?",
    "When does open enrollment close?",
    "What's the approval process for expenses?",
    "Compare the Basic, Standard, and Premium health plans.",
    "How do I apply for family medical leave?",
    "Explain the annual review cycle.",
    "How does the referral bonus program work?",
    "What is the parental leave duration?",
    "What are the password complexity requirements?",
    "How do RSUs vest for senior engineers?",
    "What's the company 401k match?",
    "Can I work from home?",
    "How do I report harassment?",
    "What is the access control policy?",
    "What errors does the API return for failed auth?",
    "What encryption is required for health data?",
    "How long must customer data be retained?",
    "What uptime does the SLA guarantee?",
    "How are code changes deployed to production?",
    "Is Acme Corp SOC 2 certified?",
    "Can enterprise customers get a refund?",
    "How do I change my password?",
    "What are the RPO and RTO targets?",
    "What happens during the 90-day probationary period?",
    "Do employees get paid for jury duty?",
    "How much PTO do I get?",
    "Benefits enrollment deadline?",
    "Who needs to sign off on my expense report?",
    "Which health plan includes dental?",
    "FMLA eligibility requirements?",
    "When are performance reviews conducted?",
    "Employee referral bonus amount?",
    "Maternity leave policy?",
    "Password rotation policy?",
    "Stock vesting cliff?",
    "401k employer contribution?",
    "Remote work requirements?",
    "Zero tolerance harassment policy?",
    "Least privilege access controls?",
    "OAuth API authentication?",
    "HIPAA technical safeguards?",
    "7-year data retention requirement?",
    "99.95% uptime SLA?",
    "Blue-green deployment process?",
    "ISO 27001 certification?",
    "45-day enterprise refund window?",
    "Self-service password reset?",
    "Disaster recovery objectives?",
    "What is the maximum vacation carryover?",
    "How does short-term disability work?",
    "What are the on-call compensation rates?",
    "How do webhooks work in the API?",
    "What is the patch management schedule?",
    "What is the cookie consent policy?",
    "How are vendors onboarded?",
    "What is the GDPR maximum fine?",
    "How does the customer deletion workflow work?",
    "What are the audit log retention periods?",
    "What MFA methods are supported?",
    "What is the laptop security policy?",
    "How does the ML platform serve models?",
    "What programming languages does the platform use?",
    "What are the code review approval rules?",
    "How does the canary rollout work?",
    "What is the on-call rotation schedule?",
    "How do I file a warranty claim?",
    "What is the return policy for online orders?",
    "What are the API rate limits?",
    "How does SSO configuration work?",
    "What is the data classification policy?",
    "How are security incidents classified?",
    "What is the BYOD enrollment process?",
    "What training is required for new hires?",
]

all_query_texts = (base_queries + paraphrase_variants)[:NUM_QUERIES]

print(f"Embedding {len(all_query_texts)} query vectors...")
queries = model.encode(all_query_texts, normalize_embeddings=True).astype("float32")
print(f"  Query embeddings shape: {queries.shape}")

# ---------------------------------------------------------------------------
# Step 6: Save everything
# ---------------------------------------------------------------------------
np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
np.save(OUTPUT_DIR / "queries.npy", queries)
np.save(OUTPUT_DIR / "chunk_metadata.npy", chunk_metadata, allow_pickle=True)
np.save(OUTPUT_DIR / "query_texts.npy", all_query_texts, allow_pickle=True)

print(f"\nSaved:")
print(f"  embeddings.npy: {embeddings.shape} ({num_real} real + {TARGET_VECTORS - num_real} augmented)")
print(f"  queries.npy:    {queries.shape} (real Acme Corp questions)")
print(f"  chunk_metadata.npy: source file and chunk index for {num_real} real chunks")
print(f"  query_texts.npy: {len(all_query_texts)} query strings")
