# idempotent_demo_modified.py -- Show re-ingestion after a file changes
import hashlib
from pathlib import Path
import shutil
import tempfile

def compute_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()

sample_dir = Path("data/corpus")
SUPPORTED = {".pdf", ".md", ".html", ".docx", ".txt"}
files = sorted(f for f in sample_dir.glob("*") if f.is_file() and f.suffix in SUPPORTED and not f.name.startswith("."))

# Build hash store from current state (simulates previous ingestion)
hash_store = {f.name: compute_file_hash(f) for f in files}

# Modify one file (append a new section to the retention policy)
policy_path = sample_dir / "data_retention_policy.md"
original_content = policy_path.read_text()
policy_path.write_text(original_content + "\n## 5. Audit Requirements\n\nAll retention actions must be logged.\n")

# Run ingestion again
print("=== Third run (one file modified) ===")
for f in files:
    current_hash = compute_file_hash(f)
    stored_hash = hash_store.get(f.name)

    if stored_hash == current_hash:
        print(f"INFO: Skipping {f.name}: unchanged (hash match)")
    else:
        print(f"INFO: Re-ingesting {f.name}: content changed")
        hash_store[f.name] = current_hash

# Restore the original file
policy_path.write_text(original_content)
