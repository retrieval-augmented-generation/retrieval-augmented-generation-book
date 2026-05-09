# idempotent_demo.py -- Demonstrate idempotent re-ingestion using file hashing
# This shows the concept without requiring a database connection.
import hashlib
from pathlib import Path

def compute_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()

sample_dir = Path("data/corpus")
SUPPORTED = {".pdf", ".md", ".html", ".docx", ".txt"}
files = sorted(f for f in sample_dir.glob("*") if f.is_file() and f.suffix in SUPPORTED and not f.name.startswith("."))

# Simulate a hash store (in production this would be in the database)
hash_store = {}

def ingest_corpus(files, hash_store, label):
    print(f"\n=== {label} ===")
    for f in files:
        current_hash = compute_file_hash(f)
        stored_hash = hash_store.get(f.name)

        if stored_hash == current_hash:
            print(f"INFO: Skipping {f.name}: unchanged (hash match)")
        else:
            if stored_hash is not None:
                print(f"INFO: Re-ingesting {f.name}: content changed")
            else:
                print(f"INFO: Ingested {f.name}: type={f.suffix[1:]}")
            hash_store[f.name] = current_hash

# First run: everything is new
ingest_corpus(files, hash_store, "First run")

# Second run: nothing changed
ingest_corpus(files, hash_store, "Second run (no changes)")
